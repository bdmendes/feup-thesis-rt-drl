use std::{cell::RefCell, rc::Rc};

use super::{
    task::{SimulatorTask, Task, TimeUnit},
    Simulator, SimulatorEvent, SimulatorJob, SimulatorJobState, SimulatorMode,
};

pub fn handle_start_event(
    task: Rc<RefCell<SimulatorTask>>,
    _time: TimeUnit,
    simulator: &mut Simulator,
) {
    let mut task = task.borrow_mut();

    // Update the time of the next arrival
    task.next_arrival += task.task.props().period;

    // Initialize the new job
    let job = simulator.jobs.get(&task.task.props().id).unwrap();
    job.borrow_mut().exec_time = if simulator.random_execution_time {
        Task::sample_execution_time(
            task.acet,
            task.bcet,
            task.task.props().wcet_h,
            &mut simulator.random_source,
            crate::generator::TimeSampleDistribution::Pert,
        )
    } else {
        task.acet
    };
    job.borrow_mut().run_time = 0;

    // Context switch or add to the queue
    if simulator.running_job.is_none()
        || job.borrow().task.borrow().custom_priority
            < simulator
                .running_job
                .as_ref()
                .unwrap()
                .borrow()
                .task
                .borrow()
                .custom_priority
    {
        context_switch(job.clone(), simulator);
    } else {
        simulator.ready_jobs_queue.push(job.clone());
    }
}

pub fn handle_end_event(
    task: Rc<RefCell<SimulatorTask>>,
    time: TimeUnit,
    simulator: &mut Simulator,
) {
    let job = simulator.jobs.get(&task.borrow().task.props().id).unwrap();

    // Push the end event to the event queue
    let end_event = Rc::new(RefCell::new(SimulatorEvent::End(
        job.borrow().task.clone(),
        time,
    )));
    simulator.event_queue.push(end_event);

    // Schedule the arrival of the next job of the same task
    let new_start_event = Rc::new(RefCell::new(SimulatorEvent::Start(
        job.borrow().task.clone(),
        std::cmp::max(
            simulator.now,
            task.borrow().next_arrival + task.borrow().task.props().period,
        ),
    )));
    simulator.event_queue.push(new_start_event.clone());
    job.borrow_mut().event = new_start_event;

    // Update runtime
    job.borrow_mut().run_time += simulator.now - simulator.last_context_switch;

    // Set running job to None
    simulator.running_job = None;

    // Budget exceedance handling
    if simulator.mode == SimulatorMode::LMode
        && job.borrow().run_time >= job.borrow().task.borrow().task.props().wcet_l
    {
        let is_ltask = matches!(job.borrow().task.borrow().task, Task::LTask(_));

        if is_ltask {
            simulator.push_event(Rc::new(RefCell::new(SimulatorEvent::TaskKill(
                task.clone(),
                simulator.now,
            ))));
        } else {
            change_mode(SimulatorMode::HMode, simulator);
        }
    }

    if simulator.ready_jobs_queue.is_empty() {
        // Idle handling
        match simulator.mode {
            SimulatorMode::LMode => (),
            SimulatorMode::HMode => change_mode(SimulatorMode::LMode, simulator),
        }
    } else {
        run_job(simulator.ready_jobs_queue.pop().unwrap(), simulator);
    }
}

fn run_job(job: Rc<RefCell<SimulatorJob>>, simulator: &mut Simulator) {
    // TODO: Right now, we are applying agent's actions immediately.
    // We should change this to apply the agent's actions at the end of the time slice.
    if job.borrow().is_agent && simulator.agent.is_some() {
        let agent = simulator.agent.take().unwrap();
        println!("Agent is running. instant={}", simulator.now);
        //  let time = time::Instant::now();
        agent.borrow_mut().activate(simulator);
        //  let elapsed = time.elapsed();
        // simulator.elapsed_times.push(elapsed);
        //let mem_stats = memory_stats().unwrap();
        // simulator
        //     .memory_usage
        //     .push((mem_stats.physical_mem, mem_stats.virtual_mem));
        simulator.agent = Some(agent);
    }

    context_switch(job, simulator);
}

fn context_switch(job: Rc<RefCell<SimulatorJob>>, simulator: &mut Simulator) {
    if let Some(running_job) = &simulator.running_job {
        // Change the state of the running_job (this is the preempted job) to READY
        running_job.borrow_mut().state = SimulatorJobState::Ready;

        // Cancel the termination event of the running_job (in the event queue)
        simulator.event_queue.retain(|event| {
            event.borrow().task().borrow().task.props().id
                != running_job.borrow().task.borrow().task.props().id
        });

        // Update the run time of the running_job
        running_job.borrow_mut().run_time += simulator.now - simulator.last_context_switch;

        // Add the running_job to the ready jobs queue
        simulator.ready_jobs_queue.push(running_job.clone());
    }

    // Push start event if the job is just starting
    if job.borrow().run_time == 0 {
        let event = Rc::new(RefCell::new(SimulatorEvent::Start(
            job.borrow().task.clone(),
            simulator.now,
        )));
        simulator.event_queue.push(event);
    }

    // Change the state of the newly arrived job to RUNNING
    job.borrow_mut().state = SimulatorJobState::Running;

    // Schedule the termination event for this job (in the event queue)
    schedule_termination_event(&mut job.borrow_mut(), simulator);

    // Update the running job to the newly arrived job
    simulator.running_job = Some(job.clone());

    simulator.last_context_switch = simulator.now;
}

fn schedule_termination_event(job: &mut SimulatorJob, simulator: &mut Simulator) {
    let termination_time = if simulator.mode == SimulatorMode::LMode
        && job.exec_time > job.task.borrow().task.props().wcet_l
    {
        simulator.now + job.task.borrow().task.props().wcet_l - job.run_time
    } else {
        simulator.now + job.exec_time - job.run_time
    };

    let event = Rc::new(RefCell::new(SimulatorEvent::End(
        job.task.clone(),
        termination_time,
    )));

    job.event = event.clone();

    simulator.event_queue.push(event);
}

fn change_mode(to_mode: SimulatorMode, simulator: &mut Simulator) {
    simulator.mode = to_mode;
    simulator.push_event(Rc::new(RefCell::new(SimulatorEvent::ModeChange(
        to_mode,
        simulator.now,
    ))));

    if simulator.mode == SimulatorMode::LMode {
        // Schedule the arrival of L-tasks.
        for task in simulator.tasks.iter() {
            if let Task::LTask(_) = task.borrow().task {
                let start_event = Rc::new(RefCell::new(SimulatorEvent::Start(
                    task.clone(),
                    std::cmp::max(
                        simulator.now,
                        task.borrow().next_arrival + task.borrow().task.props().period,
                    ),
                )));
                simulator.event_queue.push(start_event);
            }
        }
    } else {
        // Dispense with the remaining L-tasks.
        simulator
            .event_queue
            .retain(|event| !matches!(event.borrow().task().borrow().task, Task::LTask(_)));
        simulator
            .ready_jobs_queue
            .retain(|job| !matches!(job.borrow().task.borrow().task, Task::LTask(_)));
    }
}
