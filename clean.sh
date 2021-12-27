set -e # exit on error!
rm -f -- src/wavepacket/_wavepacket.cpython-38-x86_64-linux-gnu.so
rm -rf _skbuild
pip install -e .
pytest
