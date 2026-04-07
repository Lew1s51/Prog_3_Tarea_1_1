#include "tarea_header.h"

//Constructor por defecto
Tensor::Tensor() : shape(), data(nullptr), total_size(0), empty(true) {
}
//Constructor con valores 
Tensor::Tensor(const std :: vector<size_t>& shape ,const std :: vector<double >& values):
    shape(shape), total_size(1), empty(false) {
        if (shape.size() > 3 || shape.size() == 0) {
                throw invalid_argument("ERROR: el número de dimensiones debe estar entre 1 y 3");
        }
        for (size_t dim : shape) {
            total_size *= dim;
        }
        if (values.size() != total_size) {
            throw std::invalid_argument("ERROR: el número de valores no coincide con el tamaño total del tensor");
        }
        data = new double[total_size];
        for (size_t i = 0; i < total_size; ++i) {
            data[i] = values[i];
        }
    };
//Constructor de copia 
Tensor::Tensor(const Tensor& other) : shape(other.shape), total_size(other.total_size), empty(other.empty) {
    if (other.data == nullptr || other.total_size == 0) {
        // Caso: tensor vacío
        data = nullptr;
    } else {
            data = new double[total_size];
            for (size_t i = 0; i < total_size; ++i) {
                data[i] = other.data[i];
            }
        }
    };
//Operador de asignación de copia
Tensor& Tensor::operator=(const Tensor& other) {
        if (this != &other) { 
            if (data != nullptr) {
                delete[] data; // Liberar memoria actual
            }
            shape = other.shape;
            total_size = other.total_size;
            empty = other.empty;
            if (other.data == nullptr || other.total_size == 0) {
                // Caso: tensor vacío
                data = nullptr;
            } else {
                data = new double[total_size];
                for (size_t i = 0; i < total_size; ++i) {
                    data[i] = other.data[i];
                }
            }
        }
        return *this;
    };
bool broadcastable(const vector<size_t>& a, const vector<size_t>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i] && a[i] != 1 && b[i] != 1) {
            return false;
        }
    }
    return true;
}
//Constructor de movimiento
Tensor::Tensor(Tensor&& other) noexcept : shape(std::move(other.shape)), total_size(other.total_size), data(other.data), empty(other.empty) {
        other.data = nullptr; // Dejar al origen en nulo
    };
//Operador de asignación de movimiento
Tensor& Tensor::operator=(Tensor&& other) noexcept {
        if (this != &other) { 
            shape = std::move(other.shape);
            total_size = other.total_size;
            empty = other.empty;
            delete[] data; // Liberar memoria actual
            data = other.data;
            other.data = nullptr; // Dejar al origen en nulo
        }
        return *this;
    };
//Destructor
Tensor::~Tensor() {
        delete[] data;
};
//Métodos estáticos
Tensor Tensor::zeros(const vector<size_t>& shape) {
    size_t total_size = 1;
    for (size_t dim : shape) {
        total_size *= dim;
    }
    vector<double> values(total_size, 0.0);
    return Tensor(shape, values);
}
Tensor Tensor::ones(const vector<size_t>& shape) {
    size_t total_size = 1;
    for (size_t dim : shape) {
        total_size *= dim;
    }
    vector<double> values(total_size, 1.0);
    return Tensor(shape, values);
}
Tensor Tensor::random(const vector<size_t>& shape, double min_val, double max_val) {
    size_t total_size = 1;
    for (size_t dim : shape) {
        total_size *= dim;
    }
    vector<double> values(total_size);
    for (size_t i = 0; i < total_size; ++i) {
        values[i] = min_val + static_cast<double>(rand()) / RAND_MAX * (max_val - min_val);
    }
    return Tensor(shape, values);
}
Tensor Tensor::arange(double start, double end) {
    if (end <= start) {
        throw invalid_argument("El valor de 'end' debe ser mayor que 'start'");
    }
    size_t total_size = static_cast<size_t>(end - start);
    vector<double> values(total_size);
    for (size_t i = 0; i < total_size; ++i) {
        values[i] = start + i;
    }
    return Tensor({total_size}, values);
}
//Sobrecargas de operadores
Tensor Tensor::operator+(const Tensor& other) const {
    if (!broadcastable(shape, other.shape)) {
        throw invalid_argument("Shapes no compatibles para broadcasting");
    }

    vector<size_t> result_shape = shape;
    for (size_t i = 0; i < shape.size(); ++i) {
        result_shape[i] = max(shape[i], other.shape[i]);
    }

    size_t result_size = 1;
    for (auto d : result_shape) result_size *= d;

    vector<double> result_values(result_size);

    // Solo soportamos hasta 2D/3D simple 
    for (size_t i = 0; i < result_size; ++i) {
        // Índices para resultado
        size_t idx = i;

        // Convertir índice plano → multiíndice
        vector<size_t> indices(result_shape.size());
        for (int d = result_shape.size() - 1; d >= 0; --d) {
            indices[d] = idx % result_shape[d];
            idx /= result_shape[d];
        }

        size_t idx_a = 0, idx_b = 0;
        size_t stride_a = 1, stride_b = 1;

        for (int d = shape.size() - 1; d >= 0; --d) {
            size_t ia = (shape[d] == 1) ? 0 : indices[d];
            size_t ib = (other.shape[d] == 1) ? 0 : indices[d];

            idx_a += ia * stride_a;
            idx_b += ib * stride_b;

            stride_a *= shape[d];
            stride_b *= other.shape[d];
        }

        result_values[i] = data[idx_a] + other.data[idx_b];
    }

    return Tensor(result_shape, result_values);
}
Tensor Tensor::operator-(const Tensor& other) const {
    if (!broadcastable(shape, other.shape)) {
        throw invalid_argument("Shapes no compatibles para broadcasting");
    }

    vector<size_t> result_shape = shape;
    for (size_t i = 0; i < shape.size(); ++i) {
        result_shape[i] = max(shape[i], other.shape[i]);
    }

    size_t result_size = 1;
    for (auto d : result_shape) result_size *= d;

    vector<double> result_values(result_size);

    // Solo soportamos hasta 2D/3D simple 
    for (size_t i = 0; i < result_size; ++i) {
        // Índices para resultado
        size_t idx = i;

        // Convertir índice plano → multiíndice
        vector<size_t> indices(result_shape.size());
        for (int d = result_shape.size() - 1; d >= 0; --d) {
            indices[d] = idx % result_shape[d];
            idx /= result_shape[d];
        }

        size_t idx_a = 0, idx_b = 0;
        size_t stride_a = 1, stride_b = 1;

        for (int d = shape.size() - 1; d >= 0; --d) {
            size_t ia = (shape[d] == 1) ? 0 : indices[d];
            size_t ib = (other.shape[d] == 1) ? 0 : indices[d];

            idx_a += ia * stride_a;
            idx_b += ib * stride_b;

            stride_a *= shape[d];
            stride_b *= other.shape[d];
        }

        result_values[i] = data[idx_a] - other.data[idx_b];
    }

    return Tensor(result_shape, result_values);
}
Tensor Tensor::operator*(const Tensor& other) const {
    if (!broadcastable(shape, other.shape)) {
        throw invalid_argument("Shapes no compatibles para broadcasting");
    }
    vector<size_t> result_shape = shape;
    for (size_t i = 0; i < shape.size(); ++i) {
        result_shape[i] = max(shape[i], other.shape[i]);
    }
    size_t result_size = 1;
    for (auto d : result_shape) result_size *= d;
    vector<double> result_values(result_size);
    //2D O 3D
    for (size_t i = 0; i < result_size; ++i) {
        // Índices para resultado
        size_t idx = i;

        // Convertir índice plano → multiíndice
        vector<size_t> indices(result_shape.size());
        for (int d = result_shape.size() - 1; d >= 0; --d) {
            indices[d] = idx % result_shape[d];
            idx /= result_shape[d];
        }

        size_t idx_a = 0, idx_b = 0;
        size_t stride_a = 1, stride_b = 1;

        for (int d = shape.size() - 1; d >= 0; --d) {
            size_t ia = (shape[d] == 1) ? 0 : indices[d];
            size_t ib = (other.shape[d] == 1) ? 0 : indices[d];

            idx_a += ia * stride_a;
            idx_b += ib * stride_b;

            stride_a *= shape[d];
            stride_b *= other.shape[d];
        }

        result_values[i] = data[idx_a] * other.data[idx_b];
    }
    return Tensor(result_shape, result_values);
}

Tensor Tensor::operator*(double scalar) const {
    vector<double> result_values(total_size);
    for (size_t i = 0; i < total_size; ++i) {
        result_values[i] = data[i] * scalar;
    }
    return Tensor(shape, result_values);
}
Tensor Tensor::apply(const TensorTransform &transform) const {
    return transform.apply(*this);
}
//Método view
Tensor Tensor::view(const vector<size_t>& new_shape) const {
    size_t new_total_size = 1;
    for (size_t dim : new_shape) {
        new_total_size *= dim;
    }
    if (new_total_size != total_size) {
        throw invalid_argument("El nuevo tamaño del tensor total debe coincidir con el tamaño del actual");
    }
    Tensor result;
    result.shape = new_shape;
    result.total_size = total_size;
    result.data = data; // Compartir los mismos datos
    result.empty = empty;
    return result;
}
//Método unsqueeze
Tensor Tensor::unsqueeze(size_t dim) const {
    if (dim > shape.size()) {
        throw invalid_argument("La dimensión debe ser menor a el nro de dimensiones actual");
    }
    vector<size_t> new_shape = shape;
    new_shape.insert(new_shape.begin() + dim, 1); // Insertar una dimensión de tamaño 1
    return view(new_shape);
}
//Método concat, crea un nuevo tensor con memoria propia
Tensor Tensor::concat(const vector<Tensor>& tensors, size_t dim) {
    if (tensors.empty()) {
                throw invalid_argument("ERROR: La lista de tensores no debe ser vacía");
            }
            if (dim >= tensors[0].shape.size()) {
                throw invalid_argument("ERROR: La dimensión para concatenar no es válida");
            }

            // Validar compatibilidad dimensional
            size_t total_size = 0;
            vector<size_t> new_shape = tensors[0].shape;
            for (const Tensor& t : tensors) {
                if (t.shape.size() != new_shape.size()) {
                    throw invalid_argument("ERROR: Todos los tensores deben tener = nro de dimensiones");
                }
                for (size_t i = 0; i < new_shape.size(); ++i) {
                    if (i == dim) continue; // Ignorar la dimensión de concatenación
                    if (t.shape[i] != new_shape[i]) {
                        throw invalid_argument("ERROR: Los tensores no son compatibles para concatenar");
                    }
                }
                total_size += t.shape[dim];
            }
            new_shape[dim] = total_size; // Actualizamos la dimensión de concatenación

            // Reservamos nueva memoria dinámica
            vector<double> result_values(total_size);
            size_t offset = 0;
            //Copiamos los datos
            for (const Tensor& t : tensors) {
                //Calculo del tamaño a copiar
                size_t copy_size = t.shape[dim] * (total_size / new_shape[dim]);
                for (size_t i = 0; i < copy_size; ++i) {
                    result_values[offset + i] = t.get_data()[i];
                }
                offset += copy_size;
            }

            return Tensor(new_shape, result_values); // Devolver el tensor resultante usando move
}
//Función de impresión
ostream& operator<<(ostream& os, const Tensor &t) {
    os << "Tensor(shape=[";
    for (size_t i = 0; i < t.shape.size(); ++i) {
        os << t.shape[i];
        if (i < t.shape.size() - 1) {
            os << ", ";
        }
    }
    os << "])";
    return os;
}
//Función de impresión del contenido del tensor
ostream& print_tensor(ostream& os, const Tensor &t) {
    os << "Tensor(shape=[";
    for (size_t i = 0; i < t.get_shape().size(); ++i) {
        os << t.get_shape()[i];
        if (i < t.get_shape().size() - 1) {
            os << ", ";
        }
    }
    os << "], data=[";
    for (size_t i = 0; i < t.size(); ++i) {
        os << t.get_data()[i];
        if (i < t.size() - 1) {
            os << ", ";
        }
    }
    os << "])";
    return os;
}
//Función dot y matmul
Tensor dot(const Tensor& a, const Tensor& b) {
    if (a.shape != b.shape) {
        throw std::invalid_argument("Los tensores deben tener el mismo shape para el producto punto");
    }
    double result_value = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result_value += a.get_data()[i] * b.get_data()[i];
    }
    return Tensor({1}, {result_value});
}
Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.shape.size() != 2 || b.shape.size() != 2) {
        throw std::invalid_argument("Ambos tensores deben ser bidimensionales");
    }
    if (a.shape[1] != b.shape[0]) {
        throw std::invalid_argument("Nro de columnas 1er tensor debe ser = nro filas 2do tensor");
    }
    size_t m = a.shape[0];
    size_t n = a.shape[1];
    size_t p = b.shape[1];
    vector<double> result_values(m * p, 0.0);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
            for (size_t k = 0; k < n; ++k) {
                result_values[i * p + j] += a.get_data()[i * n + k] * b.get_data()[k * p + j];
            }
        }
    }
    return Tensor({m, p}, result_values);
}
