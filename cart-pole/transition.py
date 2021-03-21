class Transition:
    def __init__(self, state, action, new_state, reward, done):
        self.state = state;
        self.action = action;
        self.new_state = new_state
        self.reward = reward
        self.done = done

