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

// Função de ativação ReLU
double relu(double x){
    // A função relu retorna x se x é maior ou igual a zero, e zero caso contrário
    return x >= 0 ? x : 0;
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
        // Armazena a ativação do neurônio na camada oculta como a função relu da soma calculada
        hidden_layer[i] = relu(sum);
    }
    // Calcula as ativações da camada de saída
    for (int i = 0; i < num_outputs; i++) {
        double sum = 0;
        for (int j = 0; j < num_hidden; j++) {
            // Calcula a soma dos pesos multiplicados pelas ativações da camada oculta para cada neurônio na camada de saída
            sum += hidden_layer[j] * weights_hidden_output[j][i];
        }
        // Armazena a ativação do neurônio na camada de saída como a função relu da soma calculada
        output_layer[i] = relu(sum);
    }
}

void backpropagate(){
    // Calcular os erros da camada de saída
    double output_errors[num_outputs];
    for (int i = 0; i < num_outputs; i++) {
        output_errors[i] = target_outputs[i] - output_layer[i];
    }

    // Calcular erros de camada oculta
    double hidden_errors[num_hidden];
    for (int i = 0; i < num_hidden; i++) {
        double sum = 0;
        for (int j = 0; j < num_outputs; j++) {
            sum += output_errors[j] * weights_hidden_output[i][j];
        }
        hidden_errors[i] = sum * (hidden_layer[i] > 0 ? 1 : 0);
    }

    // Atualizar os pesos entre as camadas ocultas e de saída
    for (int i = 0; i < num_hidden; i++) {
        for (int j = 0; j < num_outputs; j++) {
            weights_hidden_output[i][j] += 0.1 * output_errors[j] * hidden_layer[i];
        }
    }

    // Atualizar os pesos entre as camadas de entrada e as ocultas
    for (int i = 0; i < num_inputs; i++) {
        for (int j = 0; j < num_hidden; j++) {
            weights_input_hidden[i][j] += 0.1 * hidden_errors[j] * inputs[i];
        }
    }
}

int main() {

    inputs[0] = 0.1;
    inputs[1] = 0.2;
    inputs[2] = 0.3;
    target_outputs[0] = 0.5;
    target_outputs[1] = 0.2;

    // Inicializar os pesos com valores aleatórios
    for (int i = 0; i < num_inputs; i++) {
        for (int j = 0; j < num_hidden; j++) {
            weights_input_hidden[i][j] = ((double)rand() / (double)RAND_MAX);
        }
    }
    for (int i = 0; i < num_hidden; i++) {
        for (int j = 0; j < num_outputs; j++) {
            weights_hidden_output[i][j] = ((double)rand() / (double)RAND_MAX);
        }
    }

    // Treinar rede
for (int i = 0; i< 1000; i++) {
    // Propagar as entradas pela rede neural
    feed_forward();
    // Realizar a retropropagação dos erros e atualizar os pesos
    backpropagate();
}

// Imprimir as saídas obtidas pela rede neural
printf("Output 1: %f\n", output_layer[0]);
printf("Output 2: %f\n", output_layer[1]);

return 0;
}


