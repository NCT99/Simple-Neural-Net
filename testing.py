import NN
import random

if __name__ == "__main__":
    network = NN.NeuralNetwork(4, [10, 10], 1, bias = True)
    for i in range(10000):
        a = []
        b = []
        for e in range(32):
            x = random.randrange(-100,100)
            y = random.randrange(-100,100)
            z = random.randrange(-100,100)
            w = random.randrange(-100,100)
            a.append([x/100, y/100, z/100, w/100])
            b.append([(((x > 0 and y > 0) or (x < 0 and y < 0)) and ((z > 0 and w > 0) or (z < 0 and w < 0)))])
        network.train(a,b)
    accuracy = 0
    for i in range(10000):
        x = random.randrange(-100,100)
        y = random.randrange(-100,100)
        z = random.randrange(-100,100)
        w = random.randrange(-100,100)
        prediction = (network.predict([x/100, y/100, z/100, w/100])[0] > 0.5) 
        accuracy += (prediction == (((x > 0 and y > 0) or (x < 0 and y < 0)) and ((z > 0 and w > 0) or (z < 0 and w < 0))))
    print(accuracy/10000)
    
    
 
