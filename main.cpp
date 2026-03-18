#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr double kSpeedOfSound = 343.0; // скорость звука (м/с) при ~20°C
constexpr int kMaxIterations = 60;
constexpr double kTolerance = 1e-8;

struct Vec3 {
    double x{};
    double y{};
    double z{};

    Vec3 operator+(const Vec3& other) const {
        return {x + other.x, y + other.y, z + other.z};
    }

    Vec3 operator-(const Vec3& other) const {
        return {x - other.x, y - other.y, z - other.z};
    }

    Vec3 operator*(double s) const {
        return {x * s, y * s, z * s};
    }
};

double norm(const Vec3& v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

Vec3 centroid(const std::array<Vec3, 4>& points) {
    Vec3 sum{};
    for (const auto& p : points) {
        sum = sum + p;
    }
    return sum * 0.25;
}

// Решает систему 3x3 A*x = b методом Гаусса-Жордана.
Vec3 solve3x3(double A[3][3], double b[3]) {
    constexpr int n = 3;

    double aug[n][n + 1] = {
        {A[0][0], A[0][1], A[0][2], b[0]},
        {A[1][0], A[1][1], A[1][2], b[1]},
        {A[2][0], A[2][1], A[2][2], b[2]}
    };

    for (int col = 0; col < n; ++col) {
        int pivot = col;
        for (int row = col + 1; row < n; ++row) {
            if (std::fabs(aug[row][col]) > std::fabs(aug[pivot][col])) {
                pivot = row;
            }
        }

        if (std::fabs(aug[pivot][col]) < 1e-12) {
            throw std::runtime_error("Вырожденная геометрия или недостаточно информации для локализации.");
        }

        if (pivot != col) {
            for (int j = col; j <= n; ++j) {
                std::swap(aug[col][j], aug[pivot][j]);
            }
        }

        const double diag = aug[col][col];
        for (int j = col; j <= n; ++j) {
            aug[col][j] /= diag;
        }

        for (int row = 0; row < n; ++row) {
            if (row == col) {
                continue;
            }
            const double factor = aug[row][col];
            for (int j = col; j <= n; ++j) {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    return {aug[0][n], aug[1][n], aug[2][n]};
}

Vec3 estimatePositionTDOA(
    const std::array<Vec3, 4>& microphones,
    const std::array<double, 4>& arrivalTime,
    double speedOfSound = kSpeedOfSound
) {
    // Используем микрофон №0 в качестве временной опоры.
    std::array<double, 3> deltaTime{};
    for (int i = 1; i < 4; ++i) {
        deltaTime[i - 1] = arrivalTime[i] - arrivalTime[0];
    }

    Vec3 x = centroid(microphones);

    for (int iter = 0; iter < kMaxIterations; ++iter) {
        double JTJ[3][3]{};
        double JTr[3]{};
        double err2 = 0.0;

        const Vec3 d0 = x - microphones[0];
        const double r0 = norm(d0);
        if (r0 < 1e-9) {
            throw std::runtime_error("Начальная точка слишком близка к опорному микрофону.");
        }

        for (int i = 1; i < 4; ++i) {
            const Vec3 di = x - microphones[i];
            const double ri = norm(di);
            if (ri < 1e-9) {
                throw std::runtime_error("Оценённая точка совпадает с положением одного из микрофонов.");
            }

            const double predictedDiff = (ri - r0) / speedOfSound;
            const double residual = predictedDiff - deltaTime[i - 1];
            err2 += residual * residual;

            // Строка якобиана: частные производные по координатам от ((ri - r0)/c)
            const Vec3 grad = {
                (di.x / ri - d0.x / r0) / speedOfSound,
                (di.y / ri - d0.y / r0) / speedOfSound,
                (di.z / ri - d0.z / r0) / speedOfSound
            };

            const double g[3] = {grad.x, grad.y, grad.z};
            for (int row = 0; row < 3; ++row) {
                JTr[row] += g[row] * residual;
                for (int col = 0; col < 3; ++col) {
                    JTJ[row][col] += g[row] * g[col];
                }
            }
        }

        constexpr double lambda = 1e-6;
        JTJ[0][0] += lambda;
        JTJ[1][1] += lambda;
        JTJ[2][2] += lambda;

        double rhs[3] = {-JTr[0], -JTr[1], -JTr[2]};
        const Vec3 step = solve3x3(JTJ, rhs);
        x = x + step;

        if (norm(step) < kTolerance || err2 < kTolerance * kTolerance) {
            break;
        }
    }

    return x;
}

} 

int main() {
    try {
        // Координаты 4 микрофонов (в метрах).
        const std::array<Vec3, 4> microphones = {
            Vec3{0.0, 0.0, 0.0},
            Vec3{0.20, 0.0, 0.0},
            Vec3{0.0, 0.20, 0.0},
            Vec3{0.0, 0.0, 0.20}
        };

        // Пример: метки времени прихода одного акустического импульса (в секундах).
        // В реальной системе они получаются методом кросс-корреляции / поиска пиков.
        const std::array<double, 4> arrivalTime = {
            0.00144210,
            0.00130420,
            0.00136503,
            0.00126964
        };

        const Vec3 pos = estimatePositionTDOA(microphones, arrivalTime);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Оценка положения цели относительно центра решётки (м):\n";

        const Vec3 center = centroid(microphones);
        const Vec3 rel = pos - center;

        std::cout << "Абсолютные координаты: X: " << pos.x << "  Y: " << pos.y << "  Z: " << pos.z << '\n';
        std::cout << "Относительные координаты: X: " << rel.x << "  Y: " << rel.y << "  Z: " << rel.z << '\n'; // относительно центра решётки
    } catch (const std::exception& ex) {
        std::cerr << "Ошибка локализации: " << ex.what() << '\n';
        return 1;
    }

    return 0;
}
