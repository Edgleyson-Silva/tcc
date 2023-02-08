#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NUM_INPUTS 2
#define NUM_HIDDEN 3
#define NUM_OUTPUTS 1

// Sigmoid
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

int main() {
    float inputs[NUM_INPUTS] = {1.0, 0.5};
    float hidden_weights[NUM_INPUTS][NUM_HIDDEN] = {{0.3, 0.4, 0.5}, {0.6, 0.7, 0.8}};
    float hidden_biases[NUM_HIDDEN] = {0.1, 0.2, 0.3};
    float hidden_outputs[NUM_HIDDEN];

    float output_weights[NUM_HIDDEN][NUM_OUTPUTS] = {{0.9}, {1.0}, {1.1}};
    float output_biases[NUM_OUTPUTS] = {0.4};
    float outputs[NUM_OUTPUTS];

    // Forward propagation
    for (int i = 0; i < NUM_HIDDEN; i++) {
        float sum = hidden_biases[i];
        for (int j = 0; j < NUM_INPUTS; j++) {
            sum += inputs[j] * hidden_weights[j][i];
        }
        hidden_outputs[i] = sigmoid(sum);
    }

    for (int i = 0; i < NUM_OUTPUTS; i++) {
        float sum = output_biases[i];
        for (int j = 0; j < NUM_HIDDEN; j++) {
            sum += hidden_outputs[j] * output_weights[j][i];
        }
        outputs[i] = sigmoid(sum);
    }

    printf("Output: %f\n", outputs[0]);

    return 0;
}
