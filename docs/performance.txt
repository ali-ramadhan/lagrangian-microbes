Some statistics on pickling and compressing with joblib.dump:

Benchmarks are from a 10,000 particle simulation across 4 processors. All at a compression level of 3.

Compression, time,       filesize per particle per iteration
none,        10 ms,      16 bytes
zlib,        105 ms,     6.3-6.4 bytes
gzip,        105 ms,     6.3-6.4 bytes
bzip2,       250-330 ms, 5.6-5.7 bytes
xz,          300-400 ms, 4.2-4.4 bytes
lzma,        300-320 ms, 4.3-4.4 bytes

Seems like zlib is a good compromise between compression time and filesize, so I went with it. It's also the default for joblib. Some of the others like LZMA also use up a lot of memory.
