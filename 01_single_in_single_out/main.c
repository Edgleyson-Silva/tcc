#include <stdio.h>
#include <stdlib.h>
#include "simple_neural_networks.h"
#include "simple_neural_networks.c"

double temperature[] = {12,23,50,-10,16};
double weight = -2;
double neural;
int i;
int length;

int main(){

    length =sizeof(temperature)/sizeof(double);
    printf("The predicted moods are: ");

    for (i = 0; i < length; i++){
        if (single_in_single_out(temperature[i],weight) > 10){
            printf("Happy\n");
        }
        if (single_in_single_out(temperature[i],weight) < 10){
            printf("Sad\n");
        }
    }
    
    return 0;
}