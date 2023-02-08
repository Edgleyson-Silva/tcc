#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// Define o número de entradas da rede neural
#define num_inputs 3

// Define o número de neurônios na camada oculta
#define num_hidden 4

// Define o número de saídas da rede neural
#define num_outputs 2

// Vetor de entradas
double inputs[num_inputs];

// Vetor de ativações da camada oculta
double hidden_layer[num_hidden];

// Vetor de ativações da camada de saída
double output_layer[num_outputs];

// Matriz de pesos dos neurônios da entrada para a camada oculta
double weights_input_hidden[num_inputs][num_hidden];

// Matriz de pesos dos neurônios da camada oculta para a saída
double weights_hidden_output[num_hidden][num_outputs];

// Vetor de saídas alvo (objetivo)
double target_outputs[num_outputs];

// Função de ativação sigmóide
double sigmoid(double x) {
    // A função sigmóide retorna 1 / (1 + exp(-x)), onde exp(-x) é a exponenciação negativa de x
    return 1.0 / (1.0 + exp(-x));
}

// Função feed_forward, que propaga as entradas pela rede neural
void feed_forward() {
    // Calcula as ativações da camada oculta
    for (int i = 0; i < num_hidden; i++) {
        double sum = 0;
        for (int j = 0; j < num_inputs; j++) {
            // Calcula a soma dos pesos multiplicados pelas entradas para cada neurônio na camada oculta
            sum += inputs[j] * weights_input_hidden[j][i];
        }
        // Armazena a ativação do neurônio na camada oculta como a função sigmóide da soma calculada
        hidden_layer[i] = sigmoid(sum);
    }

    // Calcula as ativações da camada de saída
    for (int i = 0; i < num_outputs; i++) {
        double sum = 0;
        for (int j = 0; j < num_hidden; j++) {
            // Calcula a soma dos pesos multiplicados pelas ativações da camada oculta para cada neurônio na camada de saída
            sum += hidden_layer[j] * weights_hidden_output[j][i];
        }
        // Armazena a ativação do neurônio na camada de saída como a função sigmóide da soma calculada
        output_layer[i] = sigmoid(sum);
    }
}

void backpropagate() {
    // Calculate output layer errors
    double output_errors[num_outputs];
    for (int i = 0; i < num_outputs; i++) {
        output_errors[i] = target_outputs[i] - output_layer[i];
    }

    // Calculate hidden layer errors
    double hidden_errors[num_hidden];
    for (int i = 0; i < num_hidden; i++) {
        double sum = 0;
        for (int j = 0; j < num_outputs; j++) {
            sum += output_errors[j] * weights_hidden_output[i][j];
        }
        hidden_errors[i] = sum * hidden_layer[i] * (1.0 - hidden_layer[i]);
    }

    // Update weights between hidden and output layers
    for (int i = 0; i < num_hidden; i++) {
        for (int j = 0; j < num_outputs; j++) {
            weights_hidden_output[i][j] += 0.1 * output_errors[j] * hidden_layer[i];
        }
    }

    // Update weights between input and hidden layers
    for (int i = 0; i < num_inputs; i++) {
        for (int j = 0; j < num_hidden; j++) {
            weights_input_hidden[i][j] += 0.1 * hidden_errors[j] * inputs[i];
        }
    }
}

int main() {
// Initialize inputs
    inputs[0] = 1.0;
    inputs[1] = 0.5;
    inputs[2] = 2.0;
    // Initialize target outputs
    target_outputs[0] = 0.8;
    target_outputs[1] = 0.2;

    // Initialize weights
    for (int i = 0; i < num_inputs; i++) {
        for (int j = 0; j < num_hidden; j++) {
            weights_input_hidden[i][j] = (double)rand() / RAND_MAX;
        }
    }
    for (int i = 0; i < num_hidden; i++) {
        for (int j = 0; j < num_outputs; j++) {
            weights_hidden_output[i][j] = (double)rand() / RAND_MAX;
        }
    }

    // Train the network
    for (int i = 0; i < 1000; i++) {
        feed_forward();
        backpropagate();
    }

    // Print the output layer activations
    printf("Output layer activations:\n");
    for (int i = 0; i < num_outputs; i++) {
        printf("%.4f\n", output_layer[i]);
    }

return 0;
}