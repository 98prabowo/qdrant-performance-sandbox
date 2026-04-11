.PHONY: clean, bench-hsum-neon, bench-norm-neon

clean:
	@rm -rf ./target/criterion/

bench-hsum-neon:
	@$(MAKE) clean
	@cargo bench --bench hsum_neon "Baseline" \
		&& sleep 10 \
		&& cargo bench --bench hsum_neon "Proposed" \

bench-norm-neon:
	@$(MAKE) clean
	@cargo bench --bench norm_neon "Baseline" \
		&& sleep 10 \
		&& cargo bench --bench norm_neon "Proposed" \
