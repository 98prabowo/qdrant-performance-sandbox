.PHONY: clean, bench-hsum, bench-norm

clean:
	@rm -rf ./target/criterion/

bench-hsum:
	@$(MAKE) clean
	@cargo bench --bench hsum "Baseline" \
		&& sleep 10 \
		&& cargo bench --bench hsum "Proposed" \

bench-norm:
	@$(MAKE) clean
	@cargo bench --bench norm "Baseline" \
		&& sleep 10 \
		&& cargo bench --bench norm "Proposed" \
