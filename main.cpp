#include <cmath>
#include <iostream>
#include <random>
#include <tuple>

static constexpr int MAX_EPOCH = 2500;

class BackPropagationNetwork
{
public:
    static constexpr double LEARNING_RATE = 0.5;
    static constexpr int    NEURON_NUMBER = 4;
    static constexpr int    WEIGHT_NUMBER = 3;

public:
    BackPropagationNetwork()
    {
        for (int neuronIndex = 0; neuronIndex < NEURON_NUMBER; ++neuronIndex)
            for (int weightIndex = 0; weightIndex < WEIGHT_NUMBER; ++weightIndex)
                m_weights[neuronIndex][weightIndex] = GenerateRandomNumber(-1, 1);
    }

public:
    std::tuple<double, double> Train(double x1, double x2, double y1, double y2)
    {
        double networkResult;
        double x3, x4;
        double output1, output2;

        networkResult = 1 * m_weights[0][0] + x1 * m_weights[0][1] + x2 * m_weights[0][2];
        x3            = SigmoidActivation(networkResult);

        networkResult = 1 * m_weights[1][0] + x1 * m_weights[1][1] + x2 * m_weights[1][2];
        x4            = SigmoidActivation(networkResult);

        networkResult = 1 * m_weights[2][0] + x3 * m_weights[2][1] + x4 * m_weights[2][2];
        output1       = SigmoidActivation(networkResult);

        networkResult = 1 * m_weights[3][0] + x3 * m_weights[3][1] + x4 * m_weights[3][2];
        output2       = SigmoidActivation(networkResult);

        double deltas[4];
        double v1, v2;

        deltas[3] = output2 * (1 - output2) * (y2 - output2);
        deltas[2] = output1 * (1 - output1) * (y1 - output1);
        deltas[1] = x4 * (1 - x4) * (m_weights[2][2] * deltas[2] + m_weights[3][2] * deltas[3]);
        deltas[0] = x3 * (1 - x3) * (m_weights[2][1] * deltas[2] + m_weights[3][1] * deltas[3]);

        for (int neuronIndex = 0; neuronIndex < NEURON_NUMBER; ++neuronIndex)
        {
            if (neuronIndex < 2)
                v1 = x1, v2 = x2;
            else
                v1 = x3, v2 = x4;

            m_weights[neuronIndex][0] += LEARNING_RATE * 1 * deltas[neuronIndex];
            m_weights[neuronIndex][1] += LEARNING_RATE * v1 * deltas[neuronIndex];
            m_weights[neuronIndex][2] += LEARNING_RATE * v2 * deltas[neuronIndex];
        }

        return std::make_tuple(output1, output2);
    }

    std::tuple<double, double> Predict(double x1, double x2)
    {
        double networkResult;
        double x3, x4;
        double output1, output2;

        networkResult = 1 * m_weights[0][0] + x1 * m_weights[0][1] + x2 * m_weights[0][2];
        x3            = SigmoidActivation(networkResult);

        networkResult = 1 * m_weights[1][0] + x1 * m_weights[1][1] + x2 * m_weights[1][2];
        x4            = SigmoidActivation(networkResult);

        networkResult = 1 * m_weights[2][0] + x3 * m_weights[2][1] + x4 * m_weights[2][2];
        output1       = SigmoidActivation(networkResult);

        networkResult = 1 * m_weights[3][0] + x3 * m_weights[3][1] + x4 * m_weights[3][2];
        output2       = SigmoidActivation(networkResult);

        return std::make_tuple(output1, output2);
    }

private:
    inline double GenerateRandomNumber(double minValue, double maxValue)
    {
        std::random_device               randomDevice;
        std::mt19937                     randomEngine(randomDevice());
        std::uniform_real_distribution<> uniformDistribution(minValue, maxValue);

        return uniformDistribution(randomEngine);
    }

private:
    inline double SigmoidActivation(double x)
    {
        return 1.0 / (1.0 + exp(-x));
    }

private:
    double m_weights[NEURON_NUMBER][WEIGHT_NUMBER];
};

int main(void)
{
    BackPropagationNetwork     backPropagationNetwork;
    std::tuple<double, double> result;

    for (int epoch = 0; epoch < MAX_EPOCH; ++epoch)
    {
        backPropagationNetwork.Train(0, 0, 1, 0);
        backPropagationNetwork.Train(0, 1, 0, 1);
        backPropagationNetwork.Train(1, 0, 0, 1);
        backPropagationNetwork.Train(1, 1, 1, 0);
    }

    for (double rowIndex = 0.0; rowIndex <= 1.0; rowIndex += 0.02)
        for (double colIndex = 0.0; colIndex <= 1.0; colIndex += 0.02)
        {
            result = backPropagationNetwork.Predict(colIndex, rowIndex);
            std::cout << '(' << colIndex << ", " << rowIndex << ") = " << std::get<0>(result) << ' ' << std::get<1>(result) << '\n';
        }

    return 0;
}