

///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
namespace reorder {
///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

template<typename T = double, unsigned dims = DIMS>
void plane(const std::vector<T>& u0, std::vector<T>& u, size_t batch_size, double aspect_ratio = 1) {
  /*
   * Reorder spatial data s.t. each batch covers square area in space.
   * neglect boundary cells
   * N = input size, M = M.x * M.y = output size, where M <= N
   * s^2 = square size
   */
  const auto
    N = u.size() / dims,
    Nx = (size_t) sqrt(N * aspect_ratio), // m_sqrt
    Ny = N / Nx,
    s = (size_t) sqrt(batch_size);

  const dim2 M {FLOOR(Nx, s) * s, FLOOR(Ny, s) * s};
  assert(aspect_ratio == 1.);
  assert(Nx == Ny);
  assert(M.x == M.y);

  for (size_t i = 0; i < M.x; ++i)
    for (size_t j = 0; j < M.y; ++j) {
      // define spatial 2D indices (both for target dataset u)
      dim2
        i_batch_major = {i / s, j / s},
        i_batch_minor = {i % s, j % s};
      size_t i_transpose = (i_batch_major.x * M.y + i_batch_major.y * s) * s + i_batch_minor.x * s + i_batch_minor.y;

      // for (int dim = 0; dim < DIMS; ++dim)
      //   u[Ix(i_transpose, dim)] = u0[Ix2D(i, j, dim, Ny)];

      // x[i_transpose] = x0[Ix2D_(i, j, Ny)];
      for (int dim = 0; dim < dims; ++dim)
        u[Ix_(i_transpose, dim, dims)] = u0[Ix2D_(i,j,dim, Ny, dims)];
    }
}

// void plane(const std::vector<WAVE>& x0, std::vector<WAVE>& x,
//            size_t batch_size, double aspect_ratio) {
//   const auto
//     N = u.size() / DIMS,
//     Nx = (size_t) sqrt(N * aspect_ratio), // m_sqrt
//     Ny = N / Nx,
//     s = (size_t) sqrt(batch_size);

//   const dim2 M {FLOOR(Nx, s) * s, FLOOR(Ny, s) * s};
//   assert(aspect_ratio == 1.);
//   assert(Nx == Ny);
//   assert(M.x == M.y);

//   for (size_t i = 0; i < M.x; ++i)
//     for (size_t j = 0; j < M.y; ++j) {
//       // define spatial 2D indices (both for target dataset u)
//       dim2
//         i_batch_major = {i / s, j / s},
//         i_batch_minor = {i % s, j % s};
//       size_t i_transpose = (i_batch_major.x * M.y + i_batch_major.y * s) * s + i_batch_minor.x * s + i_batch_minor.y;
//       x[i_transpose] = x0[Ix2D_(i, j, Ny)];
//     }
// }

std::vector<WAVE> inverse(const std::vector<WAVE>& y0, size_t batch_size, double aspect_ratio = 1) {
  const auto
    N = y0.size(),
    Nx = (size_t) sqrt(N * aspect_ratio), // m_sqrt
    Ny = N / Nx,
    s = (size_t) sqrt(batch_size);

  const dim2 M {FLOOR(Nx, s) * s, FLOOR(Ny, s) * s};
  auto y = y0;
  for (size_t i = 0; i < M.x; ++i)
    for (size_t j = 0; j < M.y; ++j) {
      // define spatial 2D indices (both for target dataset u)
      dim2
        i_batch_major = {i / s, j / s},
        i_batch_minor = {i % s, j % s};
      size_t i_transpose = (i_batch_major.x * M.y + i_batch_major.y * s) * s + i_batch_minor.x * s + i_batch_minor.y;
      // y[i * Ny + j] = y0[i_transpose];
      y[Ix2D_(i,j,0, Ny,1)] = y0[i_transpose];
      assert(i * Ny + j == Ix2D_(i,j,0, Ny,1));
    }
  return y;
}

///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
} // end namespace
///////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
