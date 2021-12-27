#include <cmath>
#include <tuple>
#include "wavepacket.hpp"

namespace WP{

WavePacket::WavePacket(double Ai,
                       double sigmai,
                       double ki,
                       double vpi): A{Ai},
                                    sigma{sigmai},
                                    sigma_sq{sigmai*sigmai},
                                    k{ki},
                                    vp{vpi}{};

template<>
inline Vector WavePacket::_exponent(const Vector& z) const
{
  return -z.square()/(2*sigma_sq);
};

State WavePacket::system(const State& s, double t) const
  {

   const auto& z = s[0];
   const auto& p = s[1];

   const auto dzdt = p;
   const auto dpdt = - dz(z, t);

   return State{dzdt, dpdt};

  };

};
