#pragma once

#include <cstddef>
#include <vector>
#include <stdexcept>


// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.


class Grid {
private:
  std::size_t rows_;
  std::size_t cols_;
  std::vector<double> values_;

public:
  Grid(std::size_t rows, std::size_t cols)
      : rows_(rows),
        cols_(cols),
        values_(rows * cols, 0.0) {}

  double& operator()(std::size_t i, std::size_t j){
    return values_[i * cols_ + j];
  }
  double  operator()(std::size_t i, std::size_t j) const {
    return values_[i * cols_ + j];

  }

  std::size_t rows() const {
    return rows_;
  }

  std::size_t cols() const {
    return cols_;
  }

  double* row_data(std::size_t i) noexcept {
    return values_.data() + i * cols_;
  }

  const double* row_data(std::size_t i) const noexcept{
    return values_.data() + i * cols_;
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
  #pragma omp simd
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
        "Input and output must be separate grids"
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
