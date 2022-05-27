"""
np.einsum but better

Usage:

    from better_einsum import einsum

    C = einsum("C[i,k] = A[i,j] B[j,k]", A=A, B=B)
"""

from __future__ import print_function, absolute_import, unicode_literals, division

import re
import inspect
import string
from collections import defaultdict
from itertools import zip_longest
from warnings import warn
from functools import partial
from contextlib import contextmanager

import numpy as np
from pyparsing import (
    Literal,
    Regex,
    OneOrMore,
    ZeroOrMore,
    Optional,
    StringStart,
    StringEnd,
    Group,
    ParseResults,
    ParserElement,
    replaceWith,
)


# Pyparsing setup:

ParserElement.enablePackrat()
ParserElement.setDefaultWhitespaceChars(" \t\f\n")


# Utilities:

def attach(item, action, make_copy=True):
    """Add a parse action to the given item."""
    if make_copy:
        item = item.copy()
    return item.addParseAction(action)


def regex_item(regex, options=None):
    """pyparsing.Regex except it always uses unicode."""
    if options is None:
        options = re.U
    else:
        options |= re.U
    return Regex(regex, options)


def tokenlist(item, sep, suppress=True, allow_trailing=True, at_least_two=False):
    """Create a list of tokens matching the item."""
    if suppress:
        sep = sep.suppress()
    out = item + (OneOrMore if at_least_two else ZeroOrMore)(sep + item)
    if allow_trailing:
        out += Optional(sep)
    return out


def fixto(item, output):
    """Force an item to result in a specific output."""
    return attach(item, replaceWith(output))


def parse(grammar, text):
    """Parse text using grammar."""
    tokens = grammar.parseWithTabs().parseString(text)
    if isinstance(tokens, ParseResults) and len(tokens) == 1:
        tokens = tokens[0]
    return tokens


class IndexTable(object):
    """Table of long to short index names."""

    def __init__(self):
        self.table = defaultdict(self.get_new_var)

    def get_new_var(self):
        """Get a new one-letter index name."""
        return string.ascii_letters[len(self.table)]

    def __getitem__(self, index_name):
        # if index_name is an ellipsis, pass it through unchanged
        if index_name == "...":
            return index_name
        return self.table[index_name]


# Grammar:

class EinsumGrammar(object):
    lbrack = Literal("[")
    rbrack = Literal("]")
    comma = Literal(",")
    equals = ~Literal("==") + Literal("=")
    rarrow = Literal("->") | fixto(Literal("\u2192"), "->")
    larrow = Literal("<-") | fixto(Literal("\u2190"), "<-")
    star = (
        ~Literal("**") + Literal("*")
        | fixto(Literal("\xd7"), "*")
        | fixto(Literal("\u22c5"), "*")
    )
    ellipsis = Literal("...") | fixto(Literal("\u2026"), "...")

    name = regex_item(r"(?![0-9])\w+\b")

    index = name | ellipsis
    indices = Group(Optional(tokenlist(index, comma)))

    elem = Group(
        name + Optional(lbrack.suppress() + indices + rbrack.suppress())
    )

    elem_sep = Optional(star)
    elemlist = Group(tokenlist(elem, elem_sep))

    elem_assignment = (
        (elem + (equals | larrow).suppress() + elemlist)("left assignment")
        | (elemlist + Optional(rarrow.suppress() + elem))("right assignment")
    )

    start_marker = StringStart()
    end_marker = StringEnd()

    einsum_grammar = start_marker + elem_assignment + end_marker

    @staticmethod
    def unpack_elem(elem_tokens):
        if len(elem_tokens) == 1:
            assert isinstance(elem_tokens[0], str), elem_tokens[0]
            name = elem_tokens[0]
            indices = []
        else:
            name, indices = elem_tokens
            assert isinstance(name, str), name
        return name, indices

    @classmethod
    def parse_einsum(cls, expr):
        """Parse an einsum expression."""
        tokens = parse(cls.einsum_grammar, expr)
        if len(tokens) == 1:
            raise SyntaxError("better_einsum requires explicit target indices")
        assert len(tokens) == 2, tokens

        if "left assignment" in tokens:
            assign_to, assign_from = tokens
        elif "right assignment" in tokens:
            assign_from, assign_to = tokens
        else:
            raise RuntimeError(f"invalid better_einsum grammar parse result: {tokens!r}")

        index_name_table = IndexTable()

        assign_to_name, assign_to_index_names = cls.unpack_elem(assign_to)

        assign_to_indices = "".join(index_name_table[i] for i in assign_to_index_names)

        assign_from_names = []
        assign_from_indices = []
        for assign_from_elem in assign_from:
            name, indices = cls.unpack_elem(assign_from_elem)
            assign_from_names.append(name)
            assign_from_indices.append("".join(index_name_table[i] for i in indices))

        einsum_expr = ", ".join(assign_from_indices) + " -> " + assign_to_indices
        return einsum_expr, assign_to_name, assign_from_names, index_name_table


# assign grammar elements readable names so we get better error messages
for varname, val in vars(EinsumGrammar).items():
    if isinstance(val, ParserElement):
        val.setName(varname)


def compile_einsum(expr):
    """Compile an einsum expression."""
    einsum_expr, _, _, _ = EinsumGrammar.parse_einsum(expr)
    return einsum_expr


# Base einsum:

class BaseEinsum(object):
    """Keeps track of the global einsum function to use."""

    def __init__(self, einsum):
        self.set_einsum_func(einsum)

    def set_einsum_func(self, einsum):
        """Set the base einsum function that better_einsum uses."""
        self.einsum = einsum

    @contextmanager
    def using_einsum_func(self, einsum):
        """Context manager to set teh base einsum function for better_einsum."""
        old_einsum = self.einsum
        self.set_einsum_func(einsum)
        try:
            yield
        finally:
            self.set_einsum_func(old_einsum)

    def __call__(self, *args, **kwargs):
        return self.einsum(*args, **kwargs)


base_einsum = BaseEinsum(np.einsum)


# Better einsum:

sentinel = object()


def better_einsum(expr, *given_operands, base_einsum_func=base_einsum, exec_mode=False, **kwargs):
    """The main better_einsum function.

    Supports:
    - better syntax ("C[i,k] = A[i,j] B[j,k]" instead of "ij, jk -> ik"),
    - names and indices can be arbitrary variable names not just single letters,
    - keyword arguments (einsum("C = A[i] B[i]", A=..., B=...)),
    - warnings on common bugs (e.g. if the calling scope has a different value for a variable than was passed in),
    - a .exec method for executing the einsum assignment in the calling scope, and
    - a `base_einsum_func` keyword argument for using a different base einsum function than `np.einsum`.
    """
    einsum_expr, assign_to_name, assign_from_names, index_name_table = EinsumGrammar.parse_einsum(expr)

    caller_locals = inspect.currentframe().f_back.f_locals

    operands = []
    variable_values = {}
    for name, given_operand in zip_longest(assign_from_names, given_operands, fillvalue=sentinel):
        if name is sentinel:
            raise TypeError("better_einsum received more operands than variables")

        kwargs_operand = kwargs.pop(name, sentinel)
        if kwargs_operand is not sentinel and given_operand is not sentinel:
            raise NameError(f"better_einsum got positional and keyword argument for variable: {name!r}")

        if given_operand is sentinel:
            given_operand = kwargs_operand
        found_operand = caller_locals.get(name, sentinel)

        if given_operand is sentinel and found_operand is sentinel:
            raise NameError(f"better_einsum got no value for variable: {name!r}")
        elif given_operand is sentinel:
            if not exec_mode:
                raise NameError(f"better_einsum got no value for variable: {name!r}")
            operand = found_operand
        elif found_operand is sentinel:
            operand = given_operand
        else:
            if given_operand is not found_operand:
                if exec_mode:
                    raise ValueError(f"better_einsum got two different values for the same variable {name!r}: {given_operand!r} and {found_operand!r}")
                else:
                    warn(f"better_einsum: variable {name!r} in calling scope points to a different object than was passed in; this usually denotes an error")
            operand = given_operand

        if name != "_":
            prev_value = variable_values.get(name, sentinel)
            if prev_value is not sentinel and prev_value is not operand:
                warn(f"better_einsum: got two different values for the same variable {name!r}: {prev_value!r} and {operand!r}")
            variable_values[name] = operand
        operands.append(operand)

    result = base_einsum_func(einsum_expr, *operands, **kwargs)
    if exec_mode and assign_to_name is not None:
        caller_locals[assign_to_name] = result
    return result


# Convenience aliases:

einsum = better_einsum
einsum.set_default_einsum_func = base_einsum.set_einsum_func

np_einsum = partial(einsum, base_einsum_func=np.einsum)

exec_einsum = partial(einsum, exec_mode=True)
einsum.exec = exec_einsum


# Tests:

if __name__ == "__main__":

    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    # allowed syntaxes
    assert einsum("C = A[i,j] B[i,j]", A, B) == np.sum(A * B)
    assert einsum("C[] = A[i,j] B[i,j]", A, B) == np.sum(A * B)
    assert einsum("_ = _[i,j] _[i,j]", A, B) == np.sum(A * B)
    assert einsum("arrC = arrA[ind1,ind2] arrB[ind1,ind2]", A, B) == np.sum(A * B)
    assert einsum("C = A[i,j] * B[i,j]", A, B) == np.sum(A * B)
    assert einsum("C <- A[i,j] B[i,j]", A, B) == np.sum(A * B)
    assert einsum("A[i,j] B[i,j] -> C", A, B) == np.sum(A * B)
    assert np_einsum("C = A[i,j] * B[i,j]", A, B) == np.sum(A * B)

    # most preferred form is with keyword arguments
    assert einsum("C = A[i,j] B[i,j]", A=A, B=B) == np.sum(A * B)

    # buggy; should show warnings
    assert einsum("C = A[i,j] * A[i,j]", A, B) == np.sum(A * B)

    # exec test
    einsum.exec("C = A[i,j] B[i,j]")
    assert C == np.sum(A * B)

    # test other einsum forms
    assert (einsum("C[i,k] = A[i,j] * B[j,k]", A, B) == A.dot(B)).all()
    assert (einsum("C[...] = A[i,...] B[i,...]", A, B) == np.sum(A * B, axis=0)).all()
    assert (einsum("C[...] = A[...,i] B[...,i]", A, B) == np.sum(A * B, axis=-1)).all()
