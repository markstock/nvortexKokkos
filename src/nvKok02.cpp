/*
 * nvKok02.cpp
 *
 * (c)2022 Mark J. Stock <markjstock@gmail.com>
 *
 * v0.2  simplest code, using Kokkos views instead of std::vector
 */

#include <vector>
#include <random>
#include <chrono>

#include <Kokkos_Core.hpp>

// compute using float or double
#define FLOAT float

//using floatpair = std::pair<FLOAT,FLOAT>;

// Kokkos needs reduction variables to have the + operator defined
//template <typename T,typename U>                                                   
//std::pair<T,U> operator+=(const std::pair<T,U> & l,const std::pair<T,U> & r) {   
//    return {l.first+r.first, l.second+r.second};                                    
//} 
//template <typename T,typename U>                                                   
//std::pair<T,U> operator+=(const std::pair<T,U> & r) {   
//    return {this->first+r.first, this.second+r.second};                                    
//} 

// -------------------------
// compute kernel - CPU
void nvortex_2d_nograds_cpu(
    const int32_t nSrc,
    const FLOAT* const sx,
    const FLOAT* const sy,
    const FLOAT* const ss,
    const FLOAT* const sr,
    const FLOAT tx,
    const FLOAT ty,
    const FLOAT tr,
    FLOAT* const tu,
    FLOAT* const tv) {

  // velocity accumulators for target point
  FLOAT locu = 0.0f;
  FLOAT locv = 0.0f;

  // loop over all source points
  for (int32_t j=0; j<nSrc; ++j) {
    FLOAT dx = sx[j] - tx;
    FLOAT dy = sy[j] - ty;
    FLOAT distsq = dx*dx + dy*dy + sr[j]*sr[j] + tr*tr;
    FLOAT factor = ss[j] / distsq;
    locu += dy * factor;
    locv -= dx * factor;
  }

  // save into device view
  *tu = locu / (2.0f*3.1415926536f);
  *tv = locv / (2.0f*3.1415926536f);

  return;
}

// not really alignment, just minimum block sizes
int32_t buffer(const int32_t _n, const int32_t _align) {
  // 63,64 returns 1; 64,64 returns 1; 65,64 returns 2
  return _align*(1+(_n-1)/_align);
}

// main program

static void usage() {
  fprintf(stderr, "Usage: nvKok02 [-n=<number>]\n");
  exit(1);
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);

  // number of particles/points
  int32_t npart = 100000;

  if (argc > 1) {
    if (strncmp(argv[1], "-n=", 3) == 0) {
      int num = atoi(argv[1] + 3);
      if (num < 1) usage();
      npart = num;
    }
  }

  printf( "performing 2D vortex Biot-Savart on %d points\n", npart);

  // number of GPUs present
  const int32_t ngpus = 1;
  // number of cuda streams to break work into
  const int32_t nstreams = 1;
  printf( "  ngpus ( %d )  and nstreams ( %d )\n", ngpus, nstreams);

#ifdef _OPENMP
  printf( "  max openmp threads ( %d )\n", omp_get_max_threads());
#endif

  // set stream sizes
  const int32_t nperstrm = buffer(npart/nstreams, 64);
  const int32_t npfull = nstreams*nperstrm;
  printf( "  nperstrm ( %d )  and npfull ( %d )\n", nperstrm, npfull);

  // define the host views (for now, sources and targets are the same)
  Kokkos::View<FLOAT*> hsx("host source position x", npfull);
  Kokkos::View<FLOAT*> hsy("host source position y", npfull);
  Kokkos::View<FLOAT*> hss("host source strength", npfull);
  Kokkos::View<FLOAT*> hsr("host source radius", npfull);
  Kokkos::View<FLOAT*> htu("host target velocity x", npfull);
  Kokkos::View<FLOAT*> htv("host target velocity y", npfull);

  const FLOAT thisstrmag = 1.0 / std::sqrt(npart);
  const FLOAT thisrad    = (2./3.) / std::sqrt(npart);
  //std::random_device dev;
  //std::mt19937 rng(dev());
  std::mt19937 rng(1234);
  std::uniform_real_distribution<FLOAT> xrand(0.0,1.0);
  for (int32_t i = 0; i < npart; ++i)      hsx(i) = xrand(rng);
  for (int32_t i = npart; i < npfull; ++i) hsx(i) = 0.0;
  for (int32_t i = 0; i < npart; ++i)      hsy(i) = xrand(rng);
  for (int32_t i = npart; i < npfull; ++i) hsy(i) = 0.0;
  for (int32_t i = 0; i < npart; ++i)      hss(i) = thisstrmag * (2.0*xrand(rng)-1.0);
  for (int32_t i = npart; i < npfull; ++i) hss(i) = 0.0;
  for (int32_t i = 0; i < npart; ++i)      hsr(i) = thisrad;
  for (int32_t i = npart; i < npfull; ++i) hsr(i) = thisrad;
  for (int32_t i = 0; i < npfull; ++i)     htu(i) = 0.0;
  for (int32_t i = 0; i < npfull; ++i)     htv(i) = 0.0;

  // -------------------------
  // do the CPU versions

  auto start = std::chrono::system_clock::now();

  #pragma omp parallel for
  for (int32_t i=0; i<npart; ++i) {
    nvortex_2d_nograds_cpu(npart, hsx.data(),hsy.data(),hss.data(),hsr.data(), hsx(i),hsy(i),hsr(i), &htu[i],&htv[i]);
  }

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  double time = elapsed_seconds.count();

  printf( "  host (w call) total time( %g s ) and flops( %g GFlop/s )\n", time, 1.e-9 * (double)npart*(4+14*(double)npart)/time);
  printf( "    results ( %g %g %g %g %g %g)\n", htu(0), htv(0), htu(1), htv(1), htu(npart-1), htv(npart-1));

  // and do it without the call

  start = std::chrono::system_clock::now();
  #pragma omp parallel for
  for (int32_t i=0; i<npart; ++i) {

    // velocity accumulators for target point
    FLOAT locu = 0.0f;
    FLOAT locv = 0.0f;

    // loop over all source points
    for (int32_t j=0; j<npart; ++j) {
      FLOAT dx = hsx(j) - hsx(i);
      FLOAT dy = hsy(j) - hsy(i);
      FLOAT distsq = dx*dx + dy*dy + hsr(j)*hsr(j) + hsr(i)*hsr(i);
      FLOAT factor = hss(j) / distsq;
      locu += dy * factor;
      locv -= dx * factor;
    }

    // save into device view
    htu(i) = locu / (2.0f*3.1415926536f);
    htv(i) = locv / (2.0f*3.1415926536f);
  }

  end = std::chrono::system_clock::now();
  elapsed_seconds = end-start;
  time = elapsed_seconds.count();

  printf( "  host (nested) total time( %g s ) and flops( %g GFlop/s )\n", time, 1.e-9 * (double)npart*(4+14*(double)npart)/time);
  printf( "    results ( %g %g %g %g %g %g)\n", htu(0), htv(0), htu(1), htv(1), htu(npart-1), htv(npart-1));

  // copy the results into temp vectors
  Kokkos::View<FLOAT*> htu_cpu("saved host target velocity x", npfull);
  Kokkos::View<FLOAT*> htv_cpu("saved host target velocity y", npfull);
  Kokkos::deep_copy(htu_cpu, htu);
  Kokkos::deep_copy(htv_cpu, htv);

  // -------------------------
  // do the Kokkos versions

  // set device pointers, too
  //FLOAT *dsx, *dsy, *dss, *dsr;
  //FLOAT *dtx, *dty, *dtr;
  //FLOAT *dtu, *dtv;

  start = std::chrono::system_clock::now();

  // note: default capture by reference ("[&]") only works for CPU drivers (not GPU: CUDA or HIP)
  // but View can be copied by value
  Kokkos::parallel_for ("nbody02a", npart, [=] (const int32_t i) {
    htu(i) = 0.0; htv(i) = 0.0;
    nvortex_2d_nograds_cpu(npart, hsx.data(),hsy.data(),hss.data(),hsr.data(), hsx(i),hsy(i),hsr(i), &htu[i],&htv[i]);
  } );

  // time and report
  end = std::chrono::system_clock::now();
  elapsed_seconds = end-start;
  time = elapsed_seconds.count();
  printf( "  kokkos (call) total time( %g s ) and flops( %g GFlop/s )\n", time, 1.e-9 * (double)npart*(4+14*(double)npart)/time);
  printf( "    results ( %g %g %g %g %g %g)\n", htu(0), htv(0), htu(1), htv(1), htu(npart-1), htv(npart-1));

  // do it without the call ------------------------------

  start = std::chrono::system_clock::now();

  Kokkos::parallel_for ("nbody02b",
    Kokkos::RangePolicy<>(0,npart),
    [=] (const int32_t i) {

      // velocity accumulators for target point
      FLOAT locu = 0.0f;
      FLOAT locv = 0.0f;

      // loop over all source points
      for (int32_t j=0; j<npart; ++j) {
        FLOAT dx = hsx(j) - hsx(i);
        FLOAT dy = hsy(j) - hsy(i);
        FLOAT distsq = dx*dx + dy*dy + hsr(j)*hsr(j) + hsr(i)*hsr(i);
        FLOAT factor = hss(j) / distsq;
        locu += dy * factor;
        locv -= dx * factor;
      }

      // save into device view
      htu(i) = locu / (2.0f*3.1415926536f);
      htv(i) = locv / (2.0f*3.1415926536f);
    }
  );

  // time and report
  end = std::chrono::system_clock::now();
  elapsed_seconds = end-start;
  time = elapsed_seconds.count();
  printf( "  kokkos (no call) total time( %g s ) and flops( %g GFlop/s )\n", time, 1.e-9 * (double)npart*(4+14*(double)npart)/time);
  printf( "    results ( %g %g %g %g %g %g)\n", htu(0), htv(0), htu(1), htv(1), htu(npart-1), htv(npart-1));

  // do it with hierarchical parallelism ------------------------------

  start = std::chrono::system_clock::now();

  Kokkos::parallel_for ("nbody02c",
    Kokkos::TeamPolicy<>(npart, Kokkos::AUTO),
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type & thread) {
      const int32_t i = thread.league_rank();

      // velocity accumulators for target point i
      htu(i) = 0.0f;
      htv(i) = 0.0f;
      //floatpair result(0.0,0.0);

      // loop over all source points, reduce to two variables
      // NOTE: contrary to "always name your parallel regions", you CAN'T name a nested region!!!
      // ALSO: it seems that we can't reduce two variables at a time (use View ?)
      // see: https://kokkos.github.io/kokkos-core-wiki/API/core/parallel-dispatch/parallel_reduce.html#requirements
      Kokkos::parallel_reduce (//"perpart02c",
        Kokkos::TeamThreadRange(thread, 0, npart),
        [=] (const int32_t j, FLOAT& locu/*, FLOAT& locv*/) {
        //[=] (const int32_t j, floatpair& locu) {
          FLOAT dx = hsx(j) - hsx(i);
          FLOAT dy = hsy(j) - hsy(i);
          FLOAT distsq = dx*dx + dy*dy + hsr(j)*hsr(j) + hsr(i)*hsr(i);
          FLOAT factor = hss(j) / distsq;
          locu += dy * factor;
          //locv -= dx * factor;
          //locu.first += dy * factor;
          //locu.second -= dx * factor;
        },
        //result
        htu(i)//, htv(i)
      );

      // save into device view
      htu(i) /= (2.0f*3.1415926536f);
      htv(i) /= (2.0f*3.1415926536f);
      //htu(i) = result.first / (2.0f*3.1415926536f);
      //htv(i) = result.second / (2.0f*3.1415926536f);
    }
  );

  // time and report
  end = std::chrono::system_clock::now();
  elapsed_seconds = end-start;
  time = elapsed_seconds.count();
  printf( "  kokkos (nested) total time( %g s ) and flops( %g GFlop/s )\n", time, 1.e-9 * (double)npart*(4+12*(double)npart)/time);
  printf( "    results ( %g %g %g %g %g %g)\n", htu(0), htv(0), htu(1), htv(1), htu(npart-1), htv(npart-1));

/*
  // move over all source particles first
  const int32_t srcsize = npfull*sizeof(FLOAT);
  const int32_t trgsize = npart*sizeof(FLOAT);
  hipMalloc (&dsx, srcsize);
  hipMalloc (&dsy, srcsize);
  hipMalloc (&dss, srcsize);
  hipMalloc (&dsr, srcsize);
  hipMalloc (&dtu, srcsize);
  hipMalloc (&dtv, srcsize);
  hipMemcpy (dsx, hsx.data(), srcsize, hipMemcpyHostToDevice);
  hipMemcpy (dsy, hsy.data(), srcsize, hipMemcpyHostToDevice);
  hipMemcpy (dss, hss.data(), srcsize, hipMemcpyHostToDevice);
  hipMemcpy (dsr, hsr.data(), srcsize, hipMemcpyHostToDevice);
  hipMemset (dtu, 0, trgsize);
  hipMemset (dtv, 0, trgsize);
  dtx = dsx;
  dty = dsy;
  dtr = dsr;
  hipDeviceSynchronize();

  for (int32_t nstrm=0; nstrm<nstreams; ++nstrm) {

    // round-robin the GPUs used
    //const int32_t thisgpu = nstrm % ngpus;
    //cudaSetDevice(0);

    const dim3 blocks(npfull/THREADS_PER_BLOCK, 1, 1);
    const dim3 threads(THREADS_PER_BLOCK, 1, 1);

    // move the data

    // launch the kernel
    hipLaunchKernelGGL(nvortex_2d_nograds_gpu, dim3(blocks), dim3(threads), 0, 0, nperstrm, dsx,dsy,dss,dsr, 0,dtx,dty,dtr,dtu,dtv);
    hipDeviceSynchronize();

    // check
    auto err = hipGetLastError();
    if (err != hipSuccess) {
      fprintf(stderr, "Failed to launch kernel: %s!\n", hipGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    // pull data back down
    hipMemcpy (htu.data(), dtu, trgsize, hipMemcpyDeviceToHost);
    hipMemcpy (htv.data(), dtv, trgsize, hipMemcpyDeviceToHost);
    hipDeviceSynchronize();
  }

  // join streams

  // free resources
  hipFree(dsx);
  hipFree(dsy);
  hipFree(dss);
  hipFree(dsr);
  hipFree(dtu);
  hipFree(dtv);
*/

  // compare results
  FLOAT errsum = 0.0;
  FLOAT errmax = 0.0;
  for (int32_t i=0; i<npart; ++i) {
    const FLOAT thiserr = std::pow(htu(i)-htu_cpu(i), 2) + std::pow(htv(i)-htv_cpu(i), 2);
    errsum += thiserr;
    if ((FLOAT)std::sqrt(thiserr) > errmax) {
      errmax = (FLOAT)std::sqrt(thiserr);
      //printf( "    err at %d is %g\n", i, errmax);
    }
  }
  printf( "  total host-kokkos error ( %g ) max error ( %g )\n", std::sqrt(errsum/npart), errmax);
}

