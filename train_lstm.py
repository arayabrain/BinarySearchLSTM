import lasagne
from lasagne.layers import *
from lasagne import nonlinearities
from lasagne import init

import numpy as np
import pickle
import skimage.transform
import scipy

import theano
import theano.tensor as T
import theano.sandbox.neighbours as TSN

from lasagne.utils import floatX

import matplotlib.pyplot as plt

import sys
import os
from math import exp,log

import subprocess as sp

PARAM_EXTENSION = 'params'
NETWORK = "network_lstm"

# Parameters of the model

HIDDEN = 16
OUTPUTS = 256
ROUNDS = 8

LEARN_SUPERVISED = 1e-2
LEARN_REINFORCE = 1e-2

BATCHSIZE = OUTPUTS*2

def leakyReLU(x):
	return T.maximum(x,0.04*x)

def build_model():
	net = {}

	net["targetVar"] = T.ivector('targ')
	net["reinforceWeight"] = T.vector()
	
	net["input"] = lasagne.layers.InputLayer(shape=(None,None,4) ) # Inputs are last-guess-low, last-guess-high, last guess
	net["batchsize"] = net["input"].input_var.shape[0]

	net["lstm"] = lasagne.layers.LSTMLayer(incoming = net["input"], num_units = HIDDEN, grad_clipping = 1.0)
	net["slice"] = lasagne.layers.SliceLayer( net["lstm"], -1, 1)
	net["output"] = lasagne.layers.DenseLayer(incoming = net["slice"], num_units=OUTPUTS, nonlinearity=lasagne.nonlinearities.softmax)
	
	net["getOutput"] = lasagne.layers.get_output( net["output"] )
	
	net["params"] = lasagne.layers.get_all_params( net["output"], trainable = True)
	net["loss"] = -T.log( net["getOutput"][T.arange(net["batchsize"]),net["targetVar"]] ).mean()
	net["rloss"] = -(net["reinforceWeight"] * T.log( net["getOutput"][T.arange(net["batchsize"]),net["targetVar"]] )).mean()
	
	net["updates"] = lasagne.updates.adam(net["loss"], net["params"], learning_rate = LEARN_SUPERVISED)
	net["reinforce_updates"] = lasagne.updates.adam(net["rloss"], net["params"], learning_rate = LEARN_REINFORCE)
	
	net["process"] = theano.function([net["input"].input_var], net["getOutput"])
	net["train"] = theano.function([net["input"].input_var, net["targetVar"]], [net["loss"]], updates=net["updates"])
	
	net["reinforce"] = theano.function([net["input"].input_var, net["targetVar"], net["reinforceWeight"]], [net["getOutput"]], updates=net["reinforce_updates"])
	return net

net = build_model()

def runGame(net, epoch):
	gameinputs = np.zeros( (BATCHSIZE, ROUNDS, 4) ).astype(np.float32)
	winloss = np.zeros( BATCHSIZE )
	gametargets = np.zeros( BATCHSIZE ).astype(np.int32)
	guesses = np.zeros( (BATCHSIZE, ROUNDS) )
	
	turns = 0
	avgerr = 0
	for runidx in range(BATCHSIZE):
		numbers = np.arange(0,OUTPUTS,1)
		correctidx = runidx%OUTPUTS # We play the games in order rather than randomly, for training stability
		correctnum = numbers[correctidx]
		gametargets[runidx] = correctidx
		inputs = np.zeros( (1,0,4) ).astype(np.float32)
		guess = np.zeros( 4 ).astype(np.float32)
		
		for iter in range(ROUNDS):
			inputs = np.concatenate( [inputs, guess.reshape( (1,1,4) )], axis = 1 ) 
			out = net["process"]( inputs ) 
			
			guess[0] = guess[1] = guess[2] = 0 
			
			gn = np.random.choice(numbers, p=out[0]) # Sample from the output distribution to determine the guess
						
			guess[3] = gn/float(OUTPUTS) # We want to tell the network what it guessed, but no reason to one-hot encode this
			guesses[runidx, iter] = gn
			
			if gn == correctidx: # Let the network know it guessed right
				guess[2] = 1
			else:
				if (numbers[gn] > correctnum): # High
					turns += 1
					guess[1] = 1
				else: # Low
					turns += 1
					guess[0] = 1

		gameinputs[runidx,:,:] = inputs[0,:,:] # Record the sequence of inputs to the network for this game, for later use in training
		
		# Accumulate performance statistics for the REINFORCE algorithm for each game
		winloss[runidx] = 2*out[0,correctidx]-1
		
		# Log loss of the final guess
		avgerr -= log(out[0,correctidx]+1e-16)/float(BATCHSIZE)
	
	avgwin = winloss.mean()
	wlstd = winloss.std() + 1e-3
	
	# Log the reinforcement learning parameters
	f = open("rlparams.txt","a")
	f.write("%d %.6g, %.6g\n" % (epoch, avgwin, wlstd))
	f.close()
	
	# This is the weight applied to reinforcement updates for each game
	winloss = (winloss - avgwin)/wlstd
	
	# Log a single example play sequence
	f = open("examples.txt","wb")
	for runidx in range(BATCHSIZE):
		f.write("%d  " % gametargets[runidx])
		for j in range(ROUNDS):
			f.write(" %d" % guesses[runidx,j])
		f.write("\n")
	f.close()
	
	# Make a plot of this batch of games
	plt.imshow(np.hstack([gametargets.reshape( (BATCHSIZE, 1) ), guesses]),interpolation='nearest')
	plt.axes().set_aspect(1.0/128)
	plt.savefig("training/%.6d.png" % epoch)	
	plt.clf()
	
	net["train"]( gameinputs, gametargets ) # Supervised training on the final round
	
	# Pick a random round during the game, and reinforce based on how it turned out
	movepoint = np.random.randint(ROUNDS)	
	net["reinforce"]( gameinputs[:,0:(movepoint+1),:], guesses[:,movepoint].astype(np.int32), winloss.astype(np.float32) )

	return avgerr, turns/float(BATCHSIZE)

epoch = 0
while epoch<=6000:
	err, err2 = runGame(net,epoch)
	
	# Log the error
	f = open("error.txt","a")
	f.write("%d %.6g %.6g\n" % (epoch, err, err2)) # Epoch, log-loss, average number of turns till correct guess
	f.close()

	epoch += 1
