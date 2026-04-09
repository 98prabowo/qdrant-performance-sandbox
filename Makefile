.PHONY: clean, bench-hsum, bench-norm, bench

clean:
	@rm -rf ./target/criterion/

bench-hsum:
	@$(MAKE) clean
	@cargo bench --bench hsum "Baseline" \
		&& sleep 10 \
		&& cargo bench --bench hsum "Proposed" \

bench-norm:
	@$(MAKE) clean
	@cargo bench --bench normalization "Baseline" \
		&& sleep 10 \
		&& cargo bench --bench normalization "Proposed" \

bench:
	@$(MAKE) clean
	@$(MAKE) bench-hsum
	@sleep 15
	@$(MAKE) bench-norm
