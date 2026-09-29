// Copyright 2025 D. Danopoulos, aie4ml
// SPDX-License-Identifier: Apache-2.0

#include "dense_bias_relu.h"
using namespace adf;

template<typename ConfigT>
dense_base<ConfigT>::dense_base() {
  aie::set_rounding(ConfigT::ROUNDING);
  aie::set_saturation(ConfigT::SATURATION);

  static_assert(
      ConfigT::WTS_ELEMS * sizeof(weight_t) <= 16384,
        "Weight size per tile must not exceed one AIE-ML memory bank (16 KiB)");
  static_assert(
      !ConfigT::USE_SPARSE || sizeof(weight_t) == 1 || sizeof(weight_t) == 2,
        "Sparse weight packing is implemented for 1- and 2-byte weight_t only");
  static_assert(
      !ConfigT::USE_SPARSE || (ConfigT::SPARSE_TILE_BYTES % sizeof(weight_t) == 0),
        "SPARSE_TILE_BYTES must divide evenly into weight_t elements");
  static_assert(
      !ConfigT::USE_SPARSE || (ConfigT::IN_FEAT_SLICE % ConfigT::K == 0),
        "IN_FEAT_SLICE must be divisible by the sparse microtile K");
  static_assert(
      ConfigT::IN_FEAT_SLICE % (2 * ConfigT::K) == 0,
        "IN_FEAT_SLICE must be divisible by 2*K");
  static_assert(
      ConfigT::OUT_FEAT_SLICE % (2 * ConfigT::N) == 0,
        "OUT_FEAT_SLICE must be divisible by 2*N");
  static_assert(
      ConfigT::padded_independent_extent % (2 * ConfigT::M) == 0,
      "padded_independent_extent must be divisible by 2*M");
  static_assert(
      ConfigT::padded_IN_FEAT == ConfigT::IN_FEAT_SLICE * ConfigT::CAS_LENGTH,
        "padded_IN_FEAT must equal IN_FEAT_SLICE * CAS_LENGTH");
  static_assert(
      ConfigT::PARALLELISM_CONTRACT_OUTER
          ? (ConfigT::padded_OUT_FEAT == ConfigT::OUT_FEAT_SLICE)
          : (ConfigT::padded_OUT_FEAT == ConfigT::OUT_FEAT_SLICE * ConfigT::CAS_NUM),
        "padded_OUT_FEAT must equal OUT_FEAT_SLICE * CAS_NUM ('inner') or OUT_FEAT_SLICE ('outer')");
}

template<int M, typename VT, int N>
struct row_replicator;

template<typename VT, int N>
struct row_replicator<2, VT, N> {
  static inline aie::vector<VT, 2 * N> run(const aie::vector<VT, N>& row) {
    return aie::concat(row, row);
  }
};

template<typename VT, int N>
struct row_replicator<4, VT, N> {
  static inline aie::vector<VT, 4 * N> run(const aie::vector<VT, N>& row) {
    return aie::concat(row, row, row, row);
  }
};

template<typename VT, int N>
struct row_replicator<8, VT, N> {
  static inline aie::vector<VT, 8 * N> run(const aie::vector<VT, N>& row) {
    return aie::concat(row, row, row, row, row, row, row, row);
  }
};

template<int M, typename VT, int N>
static inline aie::vector<VT, M * N>
replicate_rows(const aie::vector<VT, N>& row) {
  static_assert(M == 2 || M == 4 || M == 8, "Unsupported M; add more specializations.");
  return row_replicator<M, VT, N>::run(row);
}

// ---------------------------------------------------------------------------
// B (weight) operand feed -- the ONLY place the sparse and dense paths differ.
//
// Both specializations expose the same two-call interface:
//   b_feeder f(wts, j);   // positioned at K-tile 0 of the N-tile pair (j, j+1)
//   f.next(B0, B1);       // this K-tile's two column tiles, then advance
// so each stage's K loop below is written once and compiles for either weight layout
// (aie::mmul::mul/mac accept an aie::vector or an aie::sparse_vector alike).
//
// Only the specialization named by ConfigT::USE_SPARSE is ever instantiated, so a dense build
// never instantiates aie::sparse_vector<weight_t, ...> -- which need not even be a valid type
// for, say, a float weight_t.
// ---------------------------------------------------------------------------

template<typename ConfigT, typename MMUL, bool SPARSE = ConfigT::USE_SPARSE>
class b_feeder;

// Dense: the ROM is [K-tile][N-tile] row-major with one uncompressed MMUL::size_B tile per
// entry, so each of the pair's columns walks by one whole K-tile row.
template<typename ConfigT, typename MMUL>
class b_feeder<ConfigT, MMUL, false> {
  using weight_t = typename ConfigT::weight_t;

  static constexpr int TILE_ELEMS = (int)MMUL::size_B;
  static constexpr int KT_STRIDE  = TILE_ELEMS * (ConfigT::OUT_FEAT_SLICE / ConfigT::N);

  const weight_t* __restrict pB1;
  const weight_t* __restrict pB2;

public:
  using vec_t = aie::vector<weight_t, MMUL::size_B>;

  b_feeder(const weight_t* wts, unsigned j)
      : pB1(wts + j * TILE_ELEMS), pB2(wts + (j + 1) * TILE_ELEMS) {}

  inline void next(vec_t& B0, vec_t& B1) {
    B0 = aie::load_v<MMUL::size_B>(pB1); pB1 += KT_STRIDE;
    B1 = aie::load_v<MMUL::size_B>(pB2); pB2 += KT_STRIDE;
  }
};

// Sparse: the weight ROM was packed by pack_mmul_rhs_sparse_matrix() into the AIE-ML hardware
// sparse (2:4) format: each (K,N) microtile is compressed column-major and occupies a fixed
// ConfigT::SPARSE_TILE_BYTES (== aie::sparse_vector<weight_t, MMUL::size_B>::memory_bytes()).
// Tiles are grouped by the (j, j+1) N-tile pair this feeder serves, then by K-tile, then by the
// two columns: tile(kt=0,j), tile(kt=0,j+1), tile(kt=1,j), tile(kt=1,j+1), ...
//
// That order exists so a SINGLE live aie::sparse_vector_input_buffer_stream can serve both B
// operands, two consecutive pops per K-tile -- the same shape as aie_api's own
// gemm_int8xint8_sparse example (`tsB >> Ybuff0 >> Ybuff1`). Using one stream is load-bearing,
// not cosmetic: two concurrently-live streams (one per column) are bit-exact under
// x86simulator but WRONG under aiesimulator, because x86sim models sparse_fill() as pure
// bookkeeping (compr_bytes += 32, pointer untouched -- see
// aietools/data/aie_ml/lib/me_native.h) so inter-stream contention is invisible there, while
// on the AIE core the sparse decompression FIFO is real state the two streams corrupt for
// each other. Likewise avoid rebuilding a stream per tile: fifo_buffer_stream's constructor
// leaves its `idx_` member uninitialized while pop() branches on it. Hence the stream is a
// member, constructed exactly once per feeder, and each stage constructs exactly one feeder per
// N-tile pair -- the lifetime the verified hand-written version had. Never copy or move one.
//
// Verified bit-exact against a NumPy reference -- and against the dense kernel fed the same
// weights -- in BOTH x86simulator and aiesimulator, all 10 iterations. aie4ml compresses the
// weights it is given and never prunes, so both paths see the identical matrix. Note the
// matching L1 input DMA tiling in src/graph_plan.h tracks ConfigT::K automatically, because
// aie4ml emits both from the same resolved microtile.
template<typename ConfigT, typename MMUL>
class b_feeder<ConfigT, MMUL, true> {
  using weight_t = typename ConfigT::weight_t;

  // SPARSE_TILE_BYTES is a byte count, but wts is a weight_t*, so the stride must be in
  // weight_t elements: identical for a 1-byte weight_t, halved for a 2-byte one.
  static constexpr int TILE_ELEMS  = ConfigT::SPARSE_TILE_BYTES / (int)sizeof(weight_t);
  static constexpr int PAIR_STRIDE = TILE_ELEMS * 2 * (ConfigT::IN_FEAT_SLICE / ConfigT::K);

  aie::sparse_vector_input_buffer_stream<weight_t, MMUL::size_B> sB;

public:
  using vec_t = aie::sparse_vector<weight_t, MMUL::size_B>;

  b_feeder(const weight_t* wts, unsigned j) : sB(wts + (j / 2) * PAIR_STRIDE) {}

  b_feeder(const b_feeder&)            = delete;
  b_feeder& operator=(const b_feeder&) = delete;

  inline void next(vec_t& B0, vec_t& B1) {
    B0 = sB.pop();
    B1 = sB.pop();
  }
};


template<typename ConfigT>
void dense_single<ConfigT>::run(input_buffer<data_t>& ifm,
                                const weight_t (&wts)[ConfigT::WTS_ELEMS],
                                const bias_t (&bias)[ConfigT::OUT_FEAT_SLICE],
                                output_buffer<result_t>& ofm)
{
  static constexpr int rowA  = ConfigT::padded_independent_extent;
  static constexpr int colA  = ConfigT::IN_FEAT_SLICE;
  static constexpr int colB  = ConfigT::OUT_FEAT_SLICE;
  static constexpr int M     = ConfigT::M;
  static constexpr int K     = ConfigT::K;
  static constexpr int N     = ConfigT::N;
  static constexpr int SHIFT = ConfigT::SHIFT;

  using MMUL = aie::mmul<M, K, N, data_t, weight_t, acc_scalar_t>;
  const data_t*      pA    = ifm.data();
  const weight_t*    pB    = wts;
  const bias_t*      pBias = bias;
  result_t*          pC    = ofm.data();

  for (unsigned z = 0; z < rowA / M; z += 2) {
    result_t* __restrict pC1 = pC + (      z * (colB / N) + 0) * MMUL::size_C;
    result_t* __restrict pC2 = pC + ((z + 1) * (colB / N) + 0) * MMUL::size_C;

    for (unsigned j = 0; j < colB / N; j += 2) {
      const data_t*   __restrict pA1 = pA + (      z * (colA / K) + 0) * MMUL::size_A;
      const data_t*   __restrict pA2 = pA + ((z + 1) * (colA / K) + 0) * MMUL::size_A;

      aie::vector<data_t, MMUL::size_A> A0, A1;
      if constexpr (ConfigT::TRANSPOSE_INPUT) {
        A0 = aie::transpose(aie::load_v<MMUL::size_A>(pA1), K, M);
        A1 = aie::transpose(aie::load_v<MMUL::size_A>(pA2), K, M);
      } else {
        A0 = aie::load_v<MMUL::size_A>(pA1);
        A1 = aie::load_v<MMUL::size_A>(pA2);
      }
      pA1 += MMUL::size_A;
      pA2 += MMUL::size_A;

      MMUL C00, C01, C10, C11;

      b_feeder<ConfigT, MMUL> fB(pB, j);
      typename b_feeder<ConfigT, MMUL>::vec_t B0, B1;
      fB.next(B0, B1);

      if constexpr (ConfigT::USE_BIAS) {
        aie::vector<bias_t, N> bias_v_0 = aie::load_v<N>(pBias + j * N);
        aie::vector<bias_t, N> bias_v_1 = aie::load_v<N>(pBias + (j + 1) * N);

        auto bias_block_0 = replicate_rows<M, bias_t, N>(bias_v_0);
        auto bias_block_1 = replicate_rows<M, bias_t, N>(bias_v_1);

        C00 = bias_block_0; C00.mac(A0, B0);
        C01 = bias_block_1; C01.mac(A0, B1);
        C10 = bias_block_0; C10.mac(A1, B0);
        C11 = bias_block_1; C11.mac(A1, B1);
      } else {
        C00.mul(A0, B0);
        C01.mul(A0, B1);
        C10.mul(A1, B0);
        C11.mul(A1, B1);
      }

      for (unsigned i = 1; i < colA / K; ++i)
        chess_prepare_for_pipelining
      {
        if constexpr (ConfigT::TRANSPOSE_INPUT) {
          A0 = aie::transpose(aie::load_v<MMUL::size_A>(pA1), K, M);
          A1 = aie::transpose(aie::load_v<MMUL::size_A>(pA2), K, M);
        } else {
          A0 = aie::load_v<MMUL::size_A>(pA1);
          A1 = aie::load_v<MMUL::size_A>(pA2);
        }
        pA1 += MMUL::size_A;
        pA2 += MMUL::size_A;
        fB.next(B0, B1);

        C00.mac(A0, B0);
        C01.mac(A0, B1);
        C10.mac(A1, B0);
        C11.mac(A1, B1);
      }

      if constexpr (ConfigT::USE_RELU) {
        aie::store_v(pC1, aie::max(C00.template to_vector<result_t>(SHIFT), result_t(0))); pC1 += MMUL::size_C;
        aie::store_v(pC1, aie::max(C01.template to_vector<result_t>(SHIFT), result_t(0))); pC1 += MMUL::size_C;
        aie::store_v(pC2, aie::max(C10.template to_vector<result_t>(SHIFT), result_t(0))); pC2 += MMUL::size_C;
        aie::store_v(pC2, aie::max(C11.template to_vector<result_t>(SHIFT), result_t(0))); pC2 += MMUL::size_C;
      } else {
        aie::store_v(pC1, C00.template to_vector<result_t>(SHIFT)); pC1 += MMUL::size_C;
        aie::store_v(pC1, C01.template to_vector<result_t>(SHIFT)); pC1 += MMUL::size_C;
        aie::store_v(pC2, C10.template to_vector<result_t>(SHIFT)); pC2 += MMUL::size_C;
        aie::store_v(pC2, C11.template to_vector<result_t>(SHIFT)); pC2 += MMUL::size_C;
      }
    }
  }
}


template<typename ConfigT>
void dense_first<ConfigT>::run(input_buffer<data_t>& ifm,
                               const weight_t (&wts)[ConfigT::WTS_ELEMS],
                               output_cascade<acc_scalar_t>* outCascade)
{
  static constexpr int rowA = ConfigT::padded_independent_extent;
  static constexpr int colA = ConfigT::IN_FEAT_SLICE;
  static constexpr int colB = ConfigT::OUT_FEAT_SLICE;
  static constexpr int M    = ConfigT::M;
  static constexpr int K    = ConfigT::K;
  static constexpr int N    = ConfigT::N;

  using MMUL = aie::mmul<M, K, N, data_t, weight_t, acc_scalar_t>;
  const data_t*   pA = ifm.data();
  const weight_t* pB = wts;

  for (unsigned z = 0; z < rowA / M; z += 2) {
    for (unsigned j = 0; j < colB / N; j += 2) {
      const data_t*   __restrict pA1 = pA + (      z * (colA / K) + 0) * MMUL::size_A;
      const data_t*   __restrict pA2 = pA + ((z + 1) * (colA / K) + 0) * MMUL::size_A;

      aie::vector<data_t, MMUL::size_A> A0, A1;
      if constexpr (ConfigT::TRANSPOSE_INPUT) {
        A0 = aie::transpose(aie::load_v<MMUL::size_A>(pA1), K, M);
        A1 = aie::transpose(aie::load_v<MMUL::size_A>(pA2), K, M);
      } else {
        A0 = aie::load_v<MMUL::size_A>(pA1);
        A1 = aie::load_v<MMUL::size_A>(pA2);
      }
      pA1 += MMUL::size_A;
      pA2 += MMUL::size_A;

      MMUL C00, C01, C10, C11;

      b_feeder<ConfigT, MMUL> fB(pB, j);
      typename b_feeder<ConfigT, MMUL>::vec_t B0, B1;
      fB.next(B0, B1);

      C00.mul(A0, B0);
      C01.mul(A0, B1);
      C10.mul(A1, B0);
      C11.mul(A1, B1);

      for (unsigned i = 1; i < colA / K; ++i)
        chess_prepare_for_pipelining
      {
        if constexpr (ConfigT::TRANSPOSE_INPUT) {
          A0 = aie::transpose(aie::load_v<MMUL::size_A>(pA1), K, M);
          A1 = aie::transpose(aie::load_v<MMUL::size_A>(pA2), K, M);
        } else {
          A0 = aie::load_v<MMUL::size_A>(pA1);
          A1 = aie::load_v<MMUL::size_A>(pA2);
        }
        pA1 += MMUL::size_A;
        pA2 += MMUL::size_A;
        fB.next(B0, B1);

        C00.mac(A0, B0);
        C01.mac(A0, B1);
        C10.mac(A1, B0);
        C11.mac(A1, B1);
      }


      writeincr(outCascade, C00.to_accum());
      writeincr(outCascade, C01.to_accum());
      writeincr(outCascade, C10.to_accum());
      writeincr(outCascade, C11.to_accum());
    }
  }
}


template<typename ConfigT>
void dense_middle<ConfigT>::run(input_buffer<data_t>& ifm,
                                const weight_t (&wts)[ConfigT::WTS_ELEMS],
                                input_cascade<acc_scalar_t>* inCascade,
                                output_cascade<acc_scalar_t>* outCascade)
{
  static constexpr int rowA = ConfigT::padded_independent_extent;
  static constexpr int colA = ConfigT::IN_FEAT_SLICE;
  static constexpr int colB = ConfigT::OUT_FEAT_SLICE;
  static constexpr int M    = ConfigT::M;
  static constexpr int K    = ConfigT::K;
  static constexpr int N    = ConfigT::N;

  using MMUL = aie::mmul<M, K, N, data_t, weight_t, acc_scalar_t>;
  const data_t*   pA = ifm.data();
  const weight_t* pB = wts;

  for (unsigned z = 0; z < rowA / M; z += 2) {
    for (unsigned j = 0; j < colB / N; j += 2) {
      auto acc00 = readincr_v<MMUL::size_C>(inCascade);
      auto acc01 = readincr_v<MMUL::size_C>(inCascade);
      auto acc10 = readincr_v<MMUL::size_C>(inCascade);
      auto acc11 = readincr_v<MMUL::size_C>(inCascade);

      MMUL C00; C00 = acc00;
      MMUL C01; C01 = acc01;
      MMUL C10; C10 = acc10;
      MMUL C11; C11 = acc11;

      const data_t*   __restrict pA1 = pA + (      z * (colA / K) + 0) * MMUL::size_A;
      const data_t*   __restrict pA2 = pA + ((z + 1) * (colA / K) + 0) * MMUL::size_A;

      aie::vector<data_t, MMUL::size_A> A0, A1;
      if constexpr (ConfigT::TRANSPOSE_INPUT) {
        A0 = aie::transpose(aie::load_v<MMUL::size_A>(pA1), K, M);
        A1 = aie::transpose(aie::load_v<MMUL::size_A>(pA2), K, M);
      } else {
        A0 = aie::load_v<MMUL::size_A>(pA1);
        A1 = aie::load_v<MMUL::size_A>(pA2);
      }
      pA1 += MMUL::size_A;
      pA2 += MMUL::size_A;

      b_feeder<ConfigT, MMUL> fB(pB, j);
      typename b_feeder<ConfigT, MMUL>::vec_t B0, B1;
      fB.next(B0, B1);

      C00.mac(A0, B0);
      C01.mac(A0, B1);
      C10.mac(A1, B0);
      C11.mac(A1, B1);

      for (unsigned i = 1; i < colA / K; ++i)
        chess_prepare_for_pipelining
      {
        if constexpr (ConfigT::TRANSPOSE_INPUT) {
          A0 = aie::transpose(aie::load_v<MMUL::size_A>(pA1), K, M);
          A1 = aie::transpose(aie::load_v<MMUL::size_A>(pA2), K, M);
        } else {
          A0 = aie::load_v<MMUL::size_A>(pA1);
          A1 = aie::load_v<MMUL::size_A>(pA2);
        }
        pA1 += MMUL::size_A;
        pA2 += MMUL::size_A;
        fB.next(B0, B1);

        C00.mac(A0, B0);
        C01.mac(A0, B1);
        C10.mac(A1, B0);
        C11.mac(A1, B1);
      }


      writeincr(outCascade, C00.to_accum());
      writeincr(outCascade, C01.to_accum());
      writeincr(outCascade, C10.to_accum());
      writeincr(outCascade, C11.to_accum());
    }
  }
}


template<typename ConfigT, bool USE_BIAS, bool USE_RELU>
static inline void dense_last_impl(input_buffer<typename ConfigT::data_t>& ifm,
                                   const typename ConfigT::weight_t (&wts)[ConfigT::WTS_ELEMS],
                                   input_cascade<typename ConfigT::acc_scalar_t>* inCascade,
                                   const typename ConfigT::bias_t (&bias)[ConfigT::OUT_FEAT_SLICE],
                                   output_buffer<typename ConfigT::result_t>& ofm)
{
  using data_t       = typename ConfigT::data_t;
  using weight_t     = typename ConfigT::weight_t;
  using bias_t       = typename ConfigT::bias_t;
  using result_t     = typename ConfigT::result_t;
  using acc_scalar_t = typename ConfigT::acc_scalar_t;

  static constexpr int rowA  = ConfigT::padded_independent_extent;
  static constexpr int colA  = ConfigT::IN_FEAT_SLICE;
  static constexpr int colB  = ConfigT::OUT_FEAT_SLICE;
  static constexpr int M     = ConfigT::M;
  static constexpr int K     = ConfigT::K;
  static constexpr int N     = ConfigT::N;
  static constexpr int SHIFT = ConfigT::SHIFT;

  using MMUL = aie::mmul<M, K, N, data_t, weight_t, acc_scalar_t>;
  const data_t*      pA    = ifm.data();
  const weight_t*    pB    = wts;
  const bias_t*      pBias = bias;
  result_t*          pC    = ofm.data();

  for (unsigned z = 0; z < rowA / M; z += 2) {
    result_t* __restrict pC1 = pC + (      z * (colB / N) + 0) * MMUL::size_C;
    result_t* __restrict pC2 = pC + ((z + 1) * (colB / N) + 0) * MMUL::size_C;

    for (unsigned j = 0; j < colB / N; j += 2) {
      const data_t*   __restrict pA1 = pA + (      z * (colA / K) + 0) * MMUL::size_A;
      const data_t*   __restrict pA2 = pA + ((z + 1) * (colA / K) + 0) * MMUL::size_A;

      MMUL C00(readincr_v<MMUL::size_C>(inCascade));
      MMUL C01(readincr_v<MMUL::size_C>(inCascade));
      MMUL C10(readincr_v<MMUL::size_C>(inCascade));
      MMUL C11(readincr_v<MMUL::size_C>(inCascade));

      aie::vector<data_t, MMUL::size_A> A0, A1;
      if constexpr (ConfigT::TRANSPOSE_INPUT) {
        A0 = aie::transpose(aie::load_v<MMUL::size_A>(pA1), K, M);
        A1 = aie::transpose(aie::load_v<MMUL::size_A>(pA2), K, M);
      } else {
        A0 = aie::load_v<MMUL::size_A>(pA1);
        A1 = aie::load_v<MMUL::size_A>(pA2);
      }
      pA1 += MMUL::size_A;
      pA2 += MMUL::size_A;

      b_feeder<ConfigT, MMUL> fB(pB, j);
      typename b_feeder<ConfigT, MMUL>::vec_t B0, B1;
      fB.next(B0, B1);

      C00.mac(A0, B0);
      C01.mac(A0, B1);
      C10.mac(A1, B0);
      C11.mac(A1, B1);

      for (unsigned i = 1; i < colA / K; ++i)
        chess_prepare_for_pipelining
      {
        if constexpr (ConfigT::TRANSPOSE_INPUT) {
          A0 = aie::transpose(aie::load_v<MMUL::size_A>(pA1), K, M);
          A1 = aie::transpose(aie::load_v<MMUL::size_A>(pA2), K, M);
        } else {
          A0 = aie::load_v<MMUL::size_A>(pA1);
          A1 = aie::load_v<MMUL::size_A>(pA2);
        }
        pA1 += MMUL::size_A;
        pA2 += MMUL::size_A;
        fB.next(B0, B1);

        C00.mac(A0, B0);
        C01.mac(A0, B1);
        C10.mac(A1, B0);
        C11.mac(A1, B1);
      }


      if constexpr (USE_BIAS) {
        aie::vector<bias_t, N> bias_v_0 = aie::load_v<N>(pBias + j * N);
        aie::vector<bias_t, N> bias_v_1 = aie::load_v<N>(pBias + (j + 1) * N);

        auto bias_block_0 = replicate_rows<M, bias_t, N>(bias_v_0);
        auto bias_block_1 = replicate_rows<M, bias_t, N>(bias_v_1);

        C00 = aie::add(C00.to_accum(), bias_block_0);
        C01 = aie::add(C01.to_accum(), bias_block_1);
        C10 = aie::add(C10.to_accum(), bias_block_0);
        C11 = aie::add(C11.to_accum(), bias_block_1);
      }

      if constexpr (USE_RELU) {
        aie::store_v(pC1, aie::max(C00.template to_vector<result_t>(SHIFT), result_t(0))); pC1 += MMUL::size_C;
        aie::store_v(pC1, aie::max(C01.template to_vector<result_t>(SHIFT), result_t(0))); pC1 += MMUL::size_C;
        aie::store_v(pC2, aie::max(C10.template to_vector<result_t>(SHIFT), result_t(0))); pC2 += MMUL::size_C;
        aie::store_v(pC2, aie::max(C11.template to_vector<result_t>(SHIFT), result_t(0))); pC2 += MMUL::size_C;
      } else {
        aie::store_v(pC1, C00.template to_vector<result_t>(SHIFT)); pC1 += MMUL::size_C;
        aie::store_v(pC1, C01.template to_vector<result_t>(SHIFT)); pC1 += MMUL::size_C;
        aie::store_v(pC2, C10.template to_vector<result_t>(SHIFT)); pC2 += MMUL::size_C;
        aie::store_v(pC2, C11.template to_vector<result_t>(SHIFT)); pC2 += MMUL::size_C;
      }
    }
  }
}

//weird but might help the compiler to avoid avoids DSFG/postamble bugs?
template<typename ConfigT>
void dense_last<ConfigT>::run(input_buffer<data_t>& ifm,
                              const weight_t (&wts)[ConfigT::WTS_ELEMS],
                              input_cascade<acc_scalar_t>* inCascade,
                              const bias_t (&bias)[ConfigT::OUT_FEAT_SLICE],
                              output_buffer<result_t>& ofm)
{
  if constexpr (ConfigT::USE_BIAS) {
    if constexpr (ConfigT::USE_RELU)
      dense_last_impl<ConfigT, true,  true>(ifm, wts, inCascade, bias, ofm);
    else
      dense_last_impl<ConfigT, true,  false>(ifm, wts, inCascade, bias, ofm);
  } else {
    if constexpr (ConfigT::USE_RELU)
      dense_last_impl<ConfigT, false, true>(ifm, wts, inCascade, bias, ofm);
    else
      dense_last_impl<ConfigT, false, false>(ifm, wts, inCascade, bias, ofm);
  }
}
