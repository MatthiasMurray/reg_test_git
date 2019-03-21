import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(101)
tf.set_random_seed(101)

xrg = np.linspace(-100,100,1000)
y = np.array([x**5+x**4+x**3+x**2+x+1 for x in xrg])

#adding noise
xrg += np.random.uniform(-4,4,1000)
y += np.random.uniform(-1000,1000,1000)

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

deg3=[]

for i in range(10):
	#specify training and test sets to be used
        trainx=trains[i][0]
        trainy=trains[i][1]
        testx = tests[i][0]
        testy = tests[i][1]

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
        y_pred=tf.add(tf.multiply(X,W1),b)
        y_pred=tf.add(tf.multiply(X**2,W2),y_pred)
        y_pred=tf.add(tf.multiply(X**3,W3),y_pred)

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
