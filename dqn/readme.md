trainHistory = []
q_nnet.load(previous_saved_model)
for i -> numberOfIterations:                                    # for loop of the iteration of training
epsilon = epsilon * decayRate ** i              # decay epsilon
  	trainData = []
for episode -> numberOfEpisodes:            # for loop to self play a few games
    		state = game.init()                          # initialize game state
		result = []
    		while not end:                                 # play game until end
			q_values = q_nnet.predict(state)
    			action = epsilonGreedyAction(q_values, epsilon) 
			
result.append(state, action, q_values)
state, end = game.step(action)

for state, action, q_values in result:
	reward = calculateReward(state) # 1 for winning player, -1 for others
q_values[action] += learningRat*reward

maxQofTargetNet = max(t_nnet.predict(next_state))
q_values[action] += reward + gamma * maxQofTargetNet

trainData.append(state, q_values)
q_nnet.train(trainData + sampleOfTrainHistory)
trainHistory.extend(trainData)
if i%iterationCheck == 0:
	win_rate = evaluate(q_nnet)  # evaluate q_nnet 
	# should it compare with previous saved models? Or baselines? When to update the previous saved models?
	if win_rate > threshold:
		q_nnet.saveModel()
		break
