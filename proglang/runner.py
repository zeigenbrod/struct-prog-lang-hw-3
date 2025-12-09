#!/usr/bin/env python

import sys
from tokenizer import tokenize
from parser import parse
from evaluator import evaluate

def main():
    environment = {}
    

    #  watch=<id>

    watch_var = None
    for arg in sys.argv[2:]:
        if arg.startswith("watch="):
            watch_var = arg.split("=", 1)[1]

    # Check for command line arguments
    if len(sys.argv) > 1:
        # Filename provided, read and execute it
        with open(sys.argv[1], 'r') as f:
            source_code = f.read()
        try:
            tokens = tokenize(source_code)
            ast = parse(tokens)
            final_value, exit_status = evaluate(ast, environment, watch_var=watch_var)
            if exit_status == "exit":
                sys.exit(final_value if isinstance(final_value, int) else 0)
            # Print watch variable after execution
            if watch_var and watch_var in environment:
                print(f"watch={watch_var} {environment[watch_var]}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        # REPL loop
        while True:
            try:
                source_code = input('>> ')
                if source_code.strip() in ['exit', 'quit']:
                    break
                tokens = tokenize(source_code)
                ast = parse(tokens)
                final_value, exit_status = evaluate(ast, environment, watch_var=watch_var)
                if exit_status == "exit":
                    print(f"Exiting with code: {final_value}")
                    sys.exit(final_value if isinstance(final_value, int) else 0)
                elif final_value is not None:
                    print(final_value)
                # Always print watch variable
                if watch_var and watch_var in environment:
                    print(f"watch={watch_var} {environment[watch_var]}")
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()
