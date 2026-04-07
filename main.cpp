#include "tarea_header.h"


void prueba1 () {
    //Crear un tensor de entrada de dimensiones 1000 × 20 × 20.
    auto input = Tensor::random({1000, 20, 20}, -1.0, 1.0);
    //Imprimir el tensor de entrada
    cout << "Tinsor de entrada:" << endl;
    cout << input << endl;  
    //Transformarlo a 1000 × 400 usando view.
    auto transformado = input.view({1000, 400});
    cout << "Tensor transformado a 1000 x 400:" << endl;
    cout << transformado << endl;
    //Multiplicarlo por una matriz 400 × 100.
    auto producto = matmul(transformado, Tensor::random({400, 100}, -1.0, 1.0));
    cout << "Producto:" << endl;
    cout << producto << endl;
    //Sumar una matriz 1 × 100
    auto suma = producto + Tensor::random({1, 100}, -1.0, 1.0);
    cout << "Suma: "<<suma<< endl;
    //Aplicamos ReLU
    auto relu = suma.apply(ReLU());
    cout << "ReLU: "<<relu<< endl;
    //Multiplicar por una matriz 100 × 10
    auto producto_final = matmul(relu, Tensor::random({100, 10}, 0.0, 1.0));
    cout << "Producto final: "<<producto_final<< endl;
    //Sumar una matriz 1 × 10
    auto suma_final = producto_final + Tensor::random({1, 10}, -1.0, 1.0);
    cout << "Suma final: "<<suma_final<< endl;
    //Aplicar Sigmoid
    auto sigmoid = suma_final.apply(Sigmoid());
    cout << "Sigmoid: "<<sigmoid<< endl;    
}
void prueba_zero() {
    auto t = Tensor::zeros({2, 3});
    cout << "Tensor de ceros:" << endl;
    print_tensor(cout, t);
    cout << endl;
}   
void prueba_ones() {
    auto t = Tensor::ones({2, 3});
    cout << "Tensor de unos:" << endl;
    print_tensor(cout, t);
    cout << endl;
}
void prueba_random() {
    auto t = Tensor::random({2, 3}, -1.0, 1.0);
    cout << "Tensor aleatorio:" << endl;
    print_tensor(cout, t);
    cout << endl;
}


int main(){
//Ejecutar LA PREUBA 
    prueba1();
    //prueba_zero();
    //prueba_ones();
    //prueba_random();
    return 0;
}
