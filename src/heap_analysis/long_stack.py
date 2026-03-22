"""A helper function, for recursive functions with deep recursion.

Deeply recursive functions can hit Python's recursion limit. You can always rewrite a function to use an
explicit stack, but this involves breaking the function into multiple parts joined by an FSM.
Python's generators already do this FSM transformation for you, so you can write your function
as a generator that yields whenever it would recurse, and then use this helper to run it without hitting the recursion limit."""

def run_with_long_stack(gen):
    stack = [gen]
    next_val = None
    next_exc = None
    while stack:
        try:
            gen = stack[-1]
            if next_exc is not None:
                next_val = gen.throw(next_exc)
                next_exc = None
            else:
                next_val = gen.send(next_val)
            stack.append(next_val)
            next_val = None
        except StopIteration as e:
            stack.pop()
            next_val = e.value
        except Exception as e:
            stack.pop()
            next_exc = e
    if next_exc is not None:
        raise next_exc
    return next_val