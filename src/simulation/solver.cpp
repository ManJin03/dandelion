#include "solver.h"

#include <Eigen/Core>

using Eigen::Vector3f;

// External Force does not changed.

// Function to calculate the derivative of KineticState
KineticState derivative(const KineticState& state)
{
    return KineticState(state.velocity, state.acceleration, Eigen::Vector3f(0, 0, 0));
}

// Function to perform a single Forward Euler step
KineticState
forward_euler_step([[maybe_unused]] const KineticState& previous , const KineticState& current)
{
    Vector3f pos = current.position + current.velocity * time_step;
    Vector3f vel = current.velocity + current.acceleration * time_step;
    return {pos , vel , current.acceleration};
}

// Function to perform a single Runge-Kutta step
KineticState
runge_kutta_step([[maybe_unused]] const KineticState& previous, const KineticState& current)
{
    return current;
}

// Function to perform a single Backward Euler step
KineticState
backward_euler_step([[maybe_unused]] const KineticState& previous, const KineticState& current)
{
    return current;
}

// Function to perform a single Symplectic Euler step
KineticState symplectic_euler_step(const KineticState& previous, const KineticState& current)
{
    (void)previous;
    return current;
}
