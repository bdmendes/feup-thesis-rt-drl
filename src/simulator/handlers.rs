use std::{cell::RefCell, rc::Rc};

use crate::simulator::EndReason;

use super::{
    task::{SimulatorTask, Task, TimeUnit},
    Simulator, SimulatorEvent, SimulatorJob, SimulatorMode,
};

pub fn handle_start_event(
    task: Rc<RefCell<SimulatorTask>>,
    time: TimeUnit,
    simulator: &mut Simulator,
) {
    simulator.push_event(Rc::new(RefCell::new(SimulatorEvent::Start(
        task.clone(),
        time,
    ))));

    // Update the time of the next arrival
    let period = task.borrow().task.props().period;
    task.borrow_mut().next_arrival += period;

    // Initialize the new job
    let job = simulator.jobs.get(&task.borrow().task.props().id).unwrap();
    println!(
        "Handling start event for task: {}; instant: {}",
        job.borrow().task.borrow().task.props().id,
        time
    );
    if job.borrow().is_agent {
        println!(
            "Scheduling agent at instant: {}, id: {}",
            time,
            job.borrow().task.borrow().task.props().id
        );
    }
    let next_exec_time = if simulator.random_execution_time {
        Task::sample_execution_time(
            task.borrow().acet,
            task.borrow().bcet,
            task.borrow().task.props().wcet_h,
            &mut simulator.random_source,
            crate::generator::TimeSampleDistribution::Pert,
        )
    } else {
        task.borrow().acet
    };
    job.borrow_mut().exec_time = next_exec_time;
    job.borrow_mut().run_time = 0;

    // Context switch or add to the queue
    if simulator.running_job.is_none()
        || job.borrow().task.borrow().task.props().id
            < simulator
                .running_job
                .as_ref()
                .unwrap()
                .borrow()
                .task
                .borrow()
                .task
                .props()
                .id
    {
        context_switch(job.clone(), simulator);
    } else {
        simulator.ready_jobs_queue.push(job.clone());
        println!(
            "Pushed job to ready queue at start: {}",
            job.borrow().task.borrow().task.props().id
        );
    }
}

pub fn handle_end_event(
    task: Rc<RefCell<SimulatorTask>>,
    time: TimeUnit,
    reason: EndReason,
    simulator: &mut Simulator,
) {
    simulator.push_event(Rc::new(RefCell::new(SimulatorEvent::End(
        task.clone(),
        time,
        reason,
    ))));

    let job = simulator.jobs.get(&task.borrow().task.props().id).unwrap();
    println!(
        "Handling end event for task: {}; instant: {}",
        job.borrow().task.borrow().task.props().id,
        time
    );

    // Schedule the arrival of the next job of the same task
    let new_start_event = Rc::new(RefCell::new(SimulatorEvent::Start(
        job.borrow().task.clone(),
        std::cmp::max(simulator.now, task.borrow().next_arrival),
    )));
    simulator.event_queue.push(new_start_event.clone());
    println!(
        "Pushed start event for task: {}",
        job.borrow().task.borrow().task.props().id
    );
    job.borrow_mut().event = new_start_event;

    // Update runtime
    job.borrow_mut().run_time += simulator.now - simulator.last_context_switch;

    // Set running job to None
    simulator.running_job = None;

    // Budget exceedance handling
    if matches!(reason, EndReason::BudgetExceedance) {
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
        let job = simulator.ready_jobs_queue.pop().unwrap();
        println!(
            "Popped job from ready queue: {}",
            job.borrow().task.borrow().task.props().id
        );
        run_job(job, simulator);
    }
}

fn run_job(job: Rc<RefCell<SimulatorJob>>, simulator: &mut Simulator) {
    // TODO: Right now, we are applying agent's actions immediately.
    // We should change this to apply the agent's actions at the end of the time slice.

    // TODO: Memory usage and time usage should be updated here.

    if job.borrow().is_agent && simulator.agent.is_some() {
        let agent = simulator.agent.take().unwrap();
        println!("Agent is running. instant={}", simulator.now);
        agent.borrow_mut().activate(simulator);
        simulator.agent = Some(agent);
    }

    println!(
        "Running job: {}",
        job.borrow().task.borrow().task.props().id
    );
    context_switch(job, simulator);
}

fn context_switch(job: Rc<RefCell<SimulatorJob>>, simulator: &mut Simulator) {
    if let Some(running_job) = &simulator.running_job {
        // Cancel the termination event of the running_job (in the event queue)
        simulator.event_queue.retain(|event| {
            event.borrow().task().borrow().task.props().id
                != running_job.borrow().task.borrow().task.props().id
        });

        // Update the run time of the running_job
        running_job.borrow_mut().run_time += simulator.now - simulator.last_context_switch;

        // Add the running_job to the ready jobs queue
        simulator.ready_jobs_queue.push(running_job.clone());
        println!(
            "Pushed job to ready queue at context switch: {}",
            running_job.borrow().task.borrow().task.props().id
        );
    }

    // Schedule the termination event for this job (in the event queue)
    schedule_termination_event(&mut job.borrow_mut(), simulator);

    // Update the running job to the newly arrived job
    simulator.running_job = Some(job.clone());
    println!(
        "Context switch to job: {}",
        job.borrow().task.borrow().task.props().id
    );

    simulator.last_context_switch = simulator.now;
}

fn schedule_termination_event(job: &mut SimulatorJob, simulator: &mut Simulator) {
    let (termination_time, reason) = if simulator.mode == SimulatorMode::LMode
        && job.exec_time > job.task.borrow().task.props().wcet_l
    {
        (
            simulator.now + job.task.borrow().task.props().wcet_l - job.run_time,
            EndReason::BudgetExceedance,
        )
    } else {
        (
            simulator.now + job.exec_time - job.run_time,
            EndReason::JobCompletion,
        )
    };

    let event = Rc::new(RefCell::new(SimulatorEvent::End(
        job.task.clone(),
        termination_time,
        reason,
    )));

    job.event = event.clone();

    simulator.event_queue.push(event);
    println!(
        "Pushed end event for job: {}",
        job.task.borrow().task.props().id
    );
}

fn change_mode(to_mode: SimulatorMode, simulator: &mut Simulator) {
    println!("Changing mode to {:?}", to_mode);

    simulator.mode = to_mode;
    simulator.push_event(Rc::new(RefCell::new(SimulatorEvent::ModeChange(
        to_mode,
        simulator.now,
    ))));

    if simulator.mode == SimulatorMode::LMode {
        // Schedule the arrival of L-tasks.
        println!("Scheduling L-tasks");
        for task in simulator.tasks.iter() {
            if let Task::LTask(_) = task.borrow().task {
                let start_event = Rc::new(RefCell::new(SimulatorEvent::Start(
                    task.clone(),
                    std::cmp::max(simulator.now, task.borrow().next_arrival),
                )));
                simulator.event_queue.push(start_event);
                println!(
                    "Pushed start event for task: {}",
                    task.borrow().task.props().id
                );
            }
        }
    } else {
        // Dispense with the remaining L-tasks.
        println!("Dispensing with L-tasks");
        simulator
            .event_queue
            .retain(|event| matches!(event.borrow().task().borrow().task, Task::HTask(_)));
        simulator
            .ready_jobs_queue
            .retain(|job| matches!(job.borrow().task.borrow().task, Task::HTask(_)));
    }
}
