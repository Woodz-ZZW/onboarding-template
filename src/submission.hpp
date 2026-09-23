#pragma once

#include <cstddef>
#include <vector>
#include <stdexcept>

#include <limits>
#include <new>
#include <type_traits>


// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.

template<class T>
struct AlignedAllocator {
  using value_type = T;
  using is_always_equal = std::true_type;

  AlignedAllocator() noexcept = default;

  template<class U>
  AlignedAllocator(const AlignedAllocator<U>&) noexcept{}

  T* allocate(std::size_t count) {
    static_assert(alignof(T) <= 64);

    if(count > std::numeric_limits<std::size_t>::max()/sizeof(T)){
      throw std::bad_array_new_length();
    }

    return static_cast<T*>(
      ::operator new(
        count * sizeof(T),
        std::align_val_t{64}
      )
    );
  }

  void deallocate(T* pointer, std::size_t) noexcept {
    ::operator delete(pointer, std::align_val_t{64});
  }
};

template<class T, class U>
constexpr bool operator==(
  const AlignedAllocator<T>&,
  const AlignedAllocator<U>&

) noexcept{
  return true;
}

template<class T, class U>
constexpr bool operator!=(
  const AlignedAllocator<T>&,
  const AlignedAllocator<U>&
) noexcept {
  return false;
}

class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;
  std::size_t stride_;

  std::vector<double, AlignedAllocator<double>> values_ ;

  static std::size_t padded_stride(std::size_t cols) {
    constexpr std::size_t multiple = 64 / sizeof(double);

    if(cols > std::numeric_limits<std::size_t>::max() 
                  - (multiple -1)){
      throw std::length_error("Grid width too large");
    }

    return ((cols + multiple -1 )/multiple) * multiple;
  }

  static std::size_t storage_size(
      std::size_t rows,
      std::size_t stride
  ){
    if (stride != 0 &&
        rows > std::numeric_limits<std::size_t>::max() / stride){
          throw std::length_error("grid size too large");
        }

    return rows * stride;
  }

public:
  Grid(std::size_t rows, std::size_t cols)
      : rows_(rows),
        cols_(cols),
        stride_(padded_stride(cols)),
        values_(storage_size(rows, stride_), 0.0) {}

  double& operator()(std::size_t i, std::size_t j) noexcept {
    return values_[i * stride_ + j];
  }
  double operator()(std::size_t i, std::size_t j) const noexcept {
    return values_[i * stride_ + j];
  }
  double* row_data(std::size_t i) noexcept {
    return values_.data() + i * stride_;
  }

  const double* row_data(std::size_t i) const noexcept {
    return values_.data() + i * stride_;
  }

  std::size_t rows() const noexcept {
    return rows_;
    }

  std::size_t cols() const noexcept {
    return cols_;
  }
};  

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.

inline void update_row(
  const double* north,
  const double* center,
  const double* south,
  double* __restrict__ out,
  std::size_t cols
) noexcept {
  #ifdef _OPENMP
  #pragma omp simd aligned(north, center, south, out:64)
  #endif
    for (std::size_t j = 1; j < cols - 1; ++j) {
        out[j] =
            0.5 * center[j] +
            0.125 * (north[j] + south[j] +
                     center[j - 1] + center[j + 1]);
    }
}

inline void apply_stencil(const Grid& old_grid, Grid& new_grid) {
    // non-aliasing guarantee
    if (&old_grid == &new_grid) {
      throw std::invalid_argument(
        "Input & output must be separate grids"
      );
    }

    const std::size_t rows = old_grid.rows();
    const std::size_t cols = old_grid.cols();

    //safeguard
    if (rows == 0 || cols == 0) {
        return;
    }

    // top and bottom boundaries
    for (std::size_t j = 0; j < cols; ++j) {
        new_grid(0, j) = old_grid(0, j);
        new_grid(rows - 1, j) = old_grid(rows - 1, j);
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(rows>=128 && cols >=128)
#endif
    for (std::size_t i = 1; i < rows - 1; ++i) {
       const double* north  = old_grid.row_data(i-1);
       const double* center = old_grid.row_data(i);
       const double* south  = old_grid.row_data(i+1);
       double* out          = new_grid.row_data(i);  

       out[0] = center[0];
       out[cols-1] = center[cols - 1];

       update_row(north,center,south,out,cols);
    }
}
