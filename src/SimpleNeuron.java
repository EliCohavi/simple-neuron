public class SimpleNeuron {
    private double weight;
    private double bias;
    private double learningRate;

    public SimpleNeuron(double learningRate) {
        // Initialize weight and bias randomly
        this.weight = Math.random();
        this.bias = Math.random();
        this.learningRate = learningRate;
    }

    // Forward pass: prediction = input * weight + bias
    public double predict(double input) {
        return input * weight + bias;
    }

    // Train on a single data point
    public void train(double input, double target) {
        double prediction = predict(input);
        double error = prediction - target;
        double loss = error * error;

        // Gradients for weight and bias (derivative of loss w.r.t weight and bias)
        double dLoss_dPred = 2 * error;          // d(loss)/d(prediction)
        double dPred_dW = input;                  // d(prediction)/d(weight)
        double dPred_dB = 1;                      // d(prediction)/d(bias)

        // Gradient descent update
        weight -= learningRate * dLoss_dPred * dPred_dW;
        bias -= learningRate * dLoss_dPred * dPred_dB;

        System.out.printf("Input: %.2f, Prediction: %.2f, Target: %.2f, Loss: %.4f, Weight: %.4f, Bias: %.4f\n",
                input, prediction, target, loss, weight, bias);
    }

    public static void main(String[] args) {
        SimpleNeuron neuron = new SimpleNeuron(0.01);

        // Training data: y = 2x + 1
        double[][] trainingData = {
                {1, 3},
                {2, 5},
                {3, 7},
                {4, 9},
                {5, 11}
        };

        // Training
        int epochs = 200;
        for (int epoch = 0; epoch < epochs; epoch++) {
            System.out.println("Epoch " + (epoch + 1));
            for (double[] data : trainingData) {
                double input = data[0];
                double target = data[1];
                neuron.train(input, target);
            }
            System.out.println("------------------------");
        }
        System.out.println("Training is finished. Commencing new data predictions.");

        // Inference (New Input)
        double[] newInputs = {6, 7, 8, 9, 10};
        for (double input : newInputs) {
            double pred = neuron.predict(input);
            System.out.printf("Prediction for input %.2f: %.2f\n", input, pred);
        }
    }
}