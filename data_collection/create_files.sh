for i in {1..1000}; do dd if=/dev/zero bs=1M count=1 of=file$i; done

