#ifndef TENSOR_H
#define TENSOR_H

#include<iostream>
#include<vector>
#include<cmath>

using namespace std;

class TensorTransform; 
class Tensor {
    private:
        vector<size_t> shape; //Vecotr de guardar dimensiones, max 3 valores
        size_t total_size;
        double *data; //Memoria array dinámico
        bool empty;
    public:
        //Funciones amigas para operaciones específicas
        friend Tensor dot(const Tensor& a, const Tensor& b);
        friend Tensor matmul(const Tensor& a, const Tensor& b);
        //Constructor por defecto
        Tensor();
        //Constructores
        Tensor(const std :: vector<size_t>& shape ,const std :: vector<double >& values);
        //Constructor  de copia
        Tensor(const Tensor& other);
        //Operador de asignación de copia
        Tensor& operator=(const Tensor& other);
        //Constructor de movimiento
        Tensor(Tensor&& other) noexcept;
        //Operador de asignación de movimiento
        Tensor& operator=(Tensor&& other) noexcept;
        //Destructor
        ~Tensor();
        //Métodos estáticos
        static Tensor zeros(const vector<size_t>& shape);
        static Tensor ones(const vector<size_t>& shape);
        static Tensor random(const vector<size_t>& shape, double min_val, double max_val);
        static Tensor arange(double start, double end);
        //Sobrecargas de operadores 
        Tensor operator+(const Tensor& other) const;
        Tensor operator-(const Tensor& other) const;
        Tensor operator*(const Tensor& other) const; // Multiplicación elemento a elemento
        Tensor operator*(double scalar) const; // Multiplicación por escalar
        //Método view, cambia la forma sin copiar los datos
        Tensor view(const vector<size_t>& new_shape) const;
        //Método unsqueeze, agrega una dimensión de tamaño 1 en la posicion dada
        Tensor unsqueeze(size_t dim)const;
        //Método concat, permite unir varios tensores creando un nuevo tensor con memoria propia
        static Tensor concat(const vector<Tensor>& tensors, size_t dim);
        //Método apply, aplica una transformación a cada elemento del tensor
        Tensor apply(const TensorTransform& transform) const;
        //Para impresión 
        friend ostream& operator<<(ostream& os, const Tensor &t);
        friend ostream& print_tensor(ostream& os, const Tensor &t);
        //Métodos de acceso a miembros privados
        size_t size() const {
            return total_size;
        }
        vector<size_t> get_shape() const {
            return shape;
        }
        double* get_data() {
            return data;
        }
        const double* get_data() const {
            return data;
        }
};
class TensorTransform {
    public:
        virtual Tensor apply(const Tensor& t) const = 0;
        virtual ~TensorTransform() = default;
};
//Clase ReLU y Sigmoid heredando de TensorTransform
class ReLU : public TensorTransform {
    public:
        Tensor apply(const Tensor& t) const override {
            Tensor result = t; // Crear una copia del tensor original
            for (size_t i = 0; i < result.size(); ++i) {
                result.get_data()[i] = max(0.0, result.get_data()[i]);
            }
            return result;
        }
};
class Sigmoid : public TensorTransform {
    public:
        Tensor apply(const Tensor& t) const override {
            Tensor result = t; 
            for (size_t i = 0; i < result.size(); ++i) {
                result.get_data()[i] = 1.0 / (1.0 + exp(-result.get_data()[i]));
            }
            return result;
        }
};
#endif //TENSOR_H