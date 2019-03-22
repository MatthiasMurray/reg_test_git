import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(101)
tf.set_random_seed(101)

xrg = np.linspace(-1,1,1000)
y = np.array([x**5+x**4+x**3+x**2+x+1 for x in xrg])

#adding noise
xrg += np.random.uniform(-.01,.01,1000)
y += np.random.uniform(-.1,.1,1000)

n = len(xrg)

#Plot training data
plt.scatter(xrg,y)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Training Data")
plt.show()

#generate training sets
indices = list(range(n))
indices_train = indices
trains = []
for i in range(10):
        idx = np.random.choice(indices_train,50,False)
        indices_train = np.array([i for i in indices_train if i not in idx])
        X=np.array([xrg[i] for i in idx])
        Y=np.array([y[i] for i in idx])
        trains.append([X,Y])

#generate testing sets
indices_test = indices
tests = []
for i in range(10):
        idx = np.random.choice(indices_test,50,False)
        indices_test = np.array([i for i in indices_test if i not in idx])
        X=np.array([xrg[i] for i in idx])
        Y=np.array([y[i] for i in idx])
        tests.append([X,Y])

deg1=[]

for i in range(10):

	trainx=trains[i][0]
	trainy=trains[i][1]
	testx = tests[i][0]
	testy = tests[i][1]
	X=tf.placeholder("float")
	Y=tf.placeholder("float")

	W=tf.Variable(np.random.randn(),name="W")
	b=tf.Variable(np.random.randn(),name="b")

	learning_rate=0.1
	training_epochs=1000

	#Hypothesis
	y_pred=tf.add(tf.multiply(X,W),b)
	#MSE Cost
	cost=tf.reduce_sum(tf.pow(y_pred-Y,2))/(2*n)

	#Grad Descent Opt
	optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

	#Global Vars Init
	init=tf.global_variables_initializer()

	#Start Session
	with tf.Session() as sess:

		#Init Vars
		sess.run(init)

		#Iterate thru epochs
		for epoch in range(training_epochs):

			#Feeding each data point into optimizer using feed dict
			for (x,y) in zip(trainx,trainy):
				sess.run(optimizer, feed_dict={X:x,Y:y})

			#Display every 100 epochs
			if (epoch+1)%100==0:
				#Calculate cost
				c=sess.run(cost,feed_dict={X:trainx,Y:trainy})
				t=sess.run(cost,feed_dict={X:testx,Y:testy})
				print("Epoch",(epoch+1),": tr. cost =",c," tst. cost =",t,"W =",sess.run(W),"b =",sess.run(b))

			#Storing necessary values to be used outside of the Session
			tr_cost=sess.run(cost,feed_dict={X:trainx,Y:trainy})
			tst_cost=sess.run(cost,feed_dict={X:testx,Y:testy})
			weight=sess.run(W)
			bias=sess.run(b)
		#Calculate predictions
		predictions=tf.add(tf.multiply(weight,trainx),bias)
		print("Training cost =",tr_cost,"Test cost =",tst_cost,"Weight =",weight,"bias =",bias,'\n')
		deg1.append([tr_cost,tst_cost,weight,bias])

print(deg1)

deg3=[]

for i in range(10):
	#specify training and test sets to be used
	trainx=trains[i][0]
	n=len(trainx)
	trainy=trains[i][1]
	testx=tests[i][0]
	testy=tests[i][1]

	#initialize variables to be used
	X=tf.placeholder("float")
	Y=tf.placeholder("float")

	W1=tf.Variable(np.random.randn(),name="W1")
	W2=tf.Variable(np.random.randn(),name="W2")
	W3=tf.Variable(np.random.randn(),name="W3")
	b=tf.Variable(np.random.randn(),name="b")
	
	#parameter setting
	learning_rate=0.1
	training_epochs=1000

	#Hypothesis
	y_pred=b+X*W1+tf.pow(X,2)*W2+tf.pow(X,3)*W3
	#MSE Cost
	cost=tf.reduce_sum(tf.pow(y_pred-Y,2))/(2*n)

	#Grad Descent Opt
	optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

	#Global Vars Init
	init=tf.global_variables_initializer()

	#Start Session
	with tf.Session() as sess:

		#Init Vars
		sess.run(init)
		cost=tf.reduce_sum(tf.pow(y_pred-Y,2))/(2*n)
		#Iterate thru epochs
		for epoch in range(training_epochs):

			#Feeding each data point into optimizer using feed dict
			for (x,y) in zip(trainx,trainy):
				sess.run(optimizer, feed_dict={X:x,Y:y})
			#Display every 100 epochs
			if (epoch+1)%100==0:
				#Calculate cost
				c=sess.run(cost,feed_dict={X:trainx,Y:trainy})
				t=sess.run(cost,feed_dict={X:testx,Y:testy})
				print("Epoch",(epoch+1),": tr. cost =",c," tst. cost =",t,"W =",[sess.run(W1),sess.run(W2),sess.run(W3)],"b =",sess.run(b))

			#Storing necessary values to be used outside of the Session
			tr_cost=sess.run(cost,feed_dict={X:trainx,Y:trainy})
			tst_cost=sess.run(cost,feed_dict={X:testx,Y:testy})
			weight1=sess.run(W1)
			weight2=sess.run(W2)
			weight3=sess.run(W3)
			bias=sess.run(b)

		print("Training cost =",tr_cost,"Test cost =",tst_cost,"Weights =",[weight1,weight2,weight3],"bias =",bias,'\n')
		deg3.append([tr_cost,tst_cost,[weight1,weight2,weight3],bias])

#show generated models                
print(deg3)

deg5=[]

for i in range(10):
	#specify training and test sets to be used
	trainx=trains[i][0]
	n=len(trainx)
	trainy=trains[i][1]
	testx=tests[i][0]
	testy=tests[i][1]

	#initialize variables to be used
	X=tf.placeholder("float")
	Y=tf.placeholder("float")

	W1=tf.Variable(np.random.randn(),name="W1")
	W2=tf.Variable(np.random.randn(),name="W2")
	W3=tf.Variable(np.random.randn(),name="W3")
	W4=tf.Variable(np.random.randn(),name="W4")
	W5=tf.Variable(np.random.randn(),name="W5")
	b=tf.Variable(np.random.randn(),name="b")
	
	#parameter setting
	learning_rate=0.1
	training_epochs=1000

	#Hypothesis
	y_pred=b+X*W1+tf.pow(X,2)*W2+tf.pow(X,3)*W3+tf.pow(X,4)*W4+tf.pow(X,5)*W5
	#MSE Cost
	cost=tf.reduce_sum(tf.pow(y_pred-Y,2))/(2*n)

	#Grad Descent Opt
	optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

	#Global Vars Init
	init=tf.global_variables_initializer()

	#Start Session
	with tf.Session() as sess:

		#Init Vars
		sess.run(init)
		cost=tf.reduce_sum(tf.pow(y_pred-Y,2))/(2*n)
		#Iterate thru epochs
		for epoch in range(training_epochs):

			#Feeding each data point into optimizer using feed dict
			for (x,y) in zip(trainx,trainy):
				sess.run(optimizer, feed_dict={X:x,Y:y})
			#Display every 100 epochs
			if (epoch+1)%100==0:
				#Calculate cost
				c=sess.run(cost,feed_dict={X:trainx,Y:trainy})
				t=sess.run(cost,feed_dict={X:testx,Y:testy})
				print("Epoch",(epoch+1),": tr. cost =",c," tst. cost =",t,"W =",[sess.run(W1),sess.run(W2),sess.run(W3),sess.run(W4),sess.run(W5)],"b =",sess.run(b))

			#Storing necessary values to be used outside of the Session
			tr_cost=sess.run(cost,feed_dict={X:trainx,Y:trainy})
			tst_cost=sess.run(cost,feed_dict={X:testx,Y:testy})
			weight1=sess.run(W1)
			weight2=sess.run(W2)
			weight3=sess.run(W3)
			weight4=sess.run(W4)
			weight5=sess.run(W5)
			bias=sess.run(b)

		print("Training cost =",tr_cost,"Test cost =",tst_cost,"Weights =",[weight1,weight2,weight3,weight4,weight5],"bias =",bias,'\n')
		deg5.append([tr_cost,tst_cost,[weight1,weight2,weight3,weight4,weight5],bias])

#show generated models                
print(deg5)

deg7=[]

for i in range(10):
	#specify training and test sets to be used
	trainx=trains[i][0]
	n=len(trainx)
	trainy=trains[i][1]
	testx=tests[i][0]
	testy=tests[i][1]

	#initialize variables to be used
	X=tf.placeholder("float")
	Y=tf.placeholder("float")

	W1=tf.Variable(np.random.randn(),name="W1")
	W2=tf.Variable(np.random.randn(),name="W2")
	W3=tf.Variable(np.random.randn(),name="W3")
	W4=tf.Variable(np.random.randn(),name="W4")
	W5=tf.Variable(np.random.randn(),name="W5")
	W6=tf.Variable(np.random.randn(),name="W6")
	W7=tf.Variable(np.random.randn(),name="W7")
	b=tf.Variable(np.random.randn(),name="b")
	
	#parameter setting
	learning_rate=0.01
	training_epochs=1000

	#Hypothesis
	y_pred=b+X*W1+tf.pow(X,2)*W2+tf.pow(X,3)*W3+tf.pow(X,4)*W4+tf.pow(X,5)*W5+tf.pow(X,6)*W6+tf.pow(X,7)*W7
	#MSE Cost
	cost=tf.reduce_sum(tf.pow(y_pred-Y,2))/(2*n)

	#Grad Descent Opt
	optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

	#Global Vars Init
	init=tf.global_variables_initializer()

	#Start Session
	with tf.Session() as sess:

		#Init Vars
		sess.run(init)
		cost=tf.reduce_sum(tf.pow(y_pred-Y,2))/(2*n)
		#Iterate thru epochs
		for epoch in range(training_epochs):

			#Feeding each data point into optimizer using feed dict
			for (x,y) in zip(trainx,trainy):
				sess.run(optimizer, feed_dict={X:x,Y:y})
			#Display every 100 epochs
			if (epoch+1)%100==0:
				#Calculate cost
				c=sess.run(cost,feed_dict={X:trainx,Y:trainy})
				t=sess.run(cost,feed_dict={X:testx,Y:testy})
				print("Epoch",(epoch+1),": tr. cost =",c," tst. cost =",t,"W =",[sess.run(W1),sess.run(W2),sess.run(W3),sess.run(W4),sess.run(W5),sess.run(W6),sess.run(W7)],"b =",sess.run(b))

			#Storing necessary values to be used outside of the Session
			tr_cost=sess.run(cost,feed_dict={X:trainx,Y:trainy})
			tst_cost=sess.run(cost,feed_dict={X:testx,Y:testy})
			weight1=sess.run(W1)
			weight2=sess.run(W2)
			weight3=sess.run(W3)
			weight4=sess.run(W4)
			weight5=sess.run(W5)
			weight6=sess.run(W6)
			weight7=sess.run(W7)
			bias=sess.run(b)

		print("Training cost =",tr_cost,"Test cost =",tst_cost,"Weights =",[weight1,weight2,weight3,weight4,weight5,weight6,weight7],"bias =",bias,'\n')
		deg7.append([tr_cost,tst_cost,[weight1,weight2,weight3,weight4,weight5,weight6,weight7],bias])

#show generated models                
print(deg7)

deg9=[]

for i in range(10):
	#specify training and test sets to be used
	trainx=trains[i][0]
	n=len(trainx)
	trainy=trains[i][1]
	testx=tests[i][0]
	testy=tests[i][1]

	#initialize variables to be used
	X=tf.placeholder("float")
	Y=tf.placeholder("float")

	W1=tf.Variable(np.random.randn(),name="W1")
	W2=tf.Variable(np.random.randn(),name="W2")
	W3=tf.Variable(np.random.randn(),name="W3")
	W4=tf.Variable(np.random.randn(),name="W4")
	W5=tf.Variable(np.random.randn(),name="W5")
	W6=tf.Variable(np.random.randn(),name="W6")
	W7=tf.Variable(np.random.randn(),name="W7")
	W8=tf.Variable(np.random.randn(),name="W8")
	W9=tf.Variable(np.random.randn(),name="W9")
	b=tf.Variable(np.random.randn(),name="b")
	
	#parameter setting
	learning_rate=0.1
	training_epochs=1000

	#Hypothesis
	y_pred=b+X*W1+tf.pow(X,2)*W2+tf.pow(X,3)*W3+tf.pow(X,4)*W4+tf.pow(X,5)*W5+tf.pow(X,6)*W6+tf.pow(X,7)*W7+tf.pow(X,8)*W8+tf.pow(X,9)*W9
	#MSE Cost
	cost=tf.reduce_sum(tf.pow(y_pred-Y,2))/(2*n)

	#Grad Descent Opt
	optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

	#Global Vars Init
	init=tf.global_variables_initializer()

	#Start Session
	with tf.Session() as sess:

		#Init Vars
		sess.run(init)
		cost=tf.reduce_sum(tf.pow(y_pred-Y,2))/(2*n)
		#Iterate thru epochs
		for epoch in range(training_epochs):

			#Feeding each data point into optimizer using feed dict
			for (x,y) in zip(trainx,trainy):
				sess.run(optimizer, feed_dict={X:x,Y:y})
			#Display every 100 epochs
			if (epoch+1)%100==0:
				#Calculate cost
				c=sess.run(cost,feed_dict={X:trainx,Y:trainy})
				t=sess.run(cost,feed_dict={X:testx,Y:testy})
				print("Epoch",(epoch+1),": tr. cost =",c," tst. cost =",t,"W =",[sess.run(W1),sess.run(W2),sess.run(W3),sess.run(W4),sess.run(W5),sess.run(W6),sess.run(W7),sess.run(W8),sess.run(W9)],"b =",sess.run(b))

			#Storing necessary values to be used outside of the Session
			tr_cost=sess.run(cost,feed_dict={X:trainx,Y:trainy})
			tst_cost=sess.run(cost,feed_dict={X:testx,Y:testy})
			weight1=sess.run(W1)
			weight2=sess.run(W2)
			weight3=sess.run(W3)
			weight4=sess.run(W4)
			weight5=sess.run(W5)
			weight6=sess.run(W6)
			weight7=sess.run(W7)
			weight8=sess.run(W8)
			weight9=sess.run(W9)
			bias=sess.run(b)

		print("Training cost =",tr_cost,"Test cost =",tst_cost,"Weights =",[weight1,weight2,weight3,weight4,weight5,weight6,weight7,weight8,weight9],"bias =",bias,'\n')
		deg9.append([tr_cost,tst_cost,[weight1,weight2,weight3,weight4,weight5,weight6,weight7,weight8,weight9],bias])

#show generated models                
print(deg9)

deg=[1,3,5,7,9]
def trn(x):
	return sum(x[i][0] for i in range(len(x)))/len(x)
def tst(x):
	return sum(x[i][1] for i in range(len(x)))/len(x)
trn=[trn(deg1),trn(deg3),trn(deg5),trn(deg7),trn(deg9)]
tst=[tst(deg1),tst(deg3),tst(deg5),tst(deg7),tst(deg9)]
plt.scatter(deg,trn,'o')
plt.scatter(deg,tst,'x')
plt.show()
