

### ActorCritic
 
 [ ] change the `train` mechanism:
        1. first fetch all the episode records
        2. discount the rewards (if necessary)
        3. compute the meta trajectories (if we are using a `meta` algorithm)
        4. shuffle the data (states, next_states, advantages, meta-trajectories, rewards, ...)
        5. generate the `batches` (manual implementation of `drop_reminder` concept needed)
        6. compute losses
        7. compute gradients
        8. apply gradients
 
 [x] make the `meta-memory` Layer (LSTM) not `stateful`
        1. set the `stateful` flag equal to `False`
        2. add to the network an additional outuput with the `LSTM` states
        3. `reset` the `LSTM` states before each iteration (which is the pair (episode, batch))
 
 [ ] support `Continuos` action space

 [ ] Action Advantage (Estimates): with or without the `returns` tensor?
 
 [ ] Understand the correct order of `training` procedure respect to the `shuffle` operation
