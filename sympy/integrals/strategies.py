"""
Integration strategy pattern implementation.

This module provides an explicit strategy pattern for integration algorithms,
separating policy (which algorithms to try, in what order) from mechanism
(how each algorithm works). This replaces the previous deeply-nested if/elif
chain in _eval_integral() with a clean, declarative strategy registry.

Each strategy represents a distinct integration algorithm or heuristic. Strategies
are tried in order, with the first successful one returning a result.
"""

from __future__ import annotations

from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.numbers import oo
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Wild
from sympy.functions import Piecewise, log, piecewise_fold
from sympy.functions.elementary.complexes import sign
from sympy.polys import Poly
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.misc import filldedent


class IntegrationStrategy:
    """
    Abstract base class for integration strategies.

    Each strategy encapsulates a specific integration algorithm or heuristic.
    Strategies are tried sequentially; the first one to successfully integrate
    the function wins.
    """

    def can_apply(self, f: Expr, x: Expr, **kwargs) -> bool:
        """
        Check if this strategy can be applied to the integral.

        Parameters
        ==========
        f : Expr
            The integrand function
        x : Expr
            The integration variable
        **kwargs :
            Additional context (eval_kwargs, conds, etc.)

        Returns
        =======
        bool
            True if this strategy should be attempted, False otherwise
        """
        return True

    def try_integrate(self, f: Expr, x: Expr, **kwargs) -> Expr | None:
        """
        Attempt to integrate the function.

        Parameters
        ==========
        f : Expr
            The integrand function
        x : Expr
            The integration variable
        **kwargs :
            Additional context (eval_kwargs, conds, etc.)

        Returns
        =======
        Expr | None
            The antiderivative if successful, None if this strategy cannot
            integrate this particular function
        """
        raise NotImplementedError


class ConstantTermStrategy(IntegrationStrategy):
    """Integrate constant terms: g(x) = const, result is c*x"""

    def can_apply(self, f: Expr, x: Expr, **kwargs) -> bool:
        meijerg = kwargs.get('meijerg', None)
        return f is S.One and not meijerg

    def try_integrate(self, f: Expr, x: Expr, **kwargs) -> Expr | None:
        return x


class PowerStrategy(IntegrationStrategy):
    """
    Integrate power terms: g(x) = (a*x + b)^c

    Handles special cases like (a*x+b)^(-1) -> log(a*x+b)
    """

    def can_apply(self, f: Expr, x: Expr, **kwargs) -> bool:
        meijerg = kwargs.get('meijerg', None)
        if meijerg or not f.is_Pow or f.exp.has(x):
            return False
        a = Wild('a', exclude=[x])
        b = Wild('b', exclude=[x])
        M = f.base.match(a*x + b)
        return M is not None

    def try_integrate(self, f: Expr, x: Expr, **kwargs) -> Expr | None:
        conds = kwargs.get('conds', 'piecewise')
        a = Wild('a', exclude=[x])
        b = Wild('b', exclude=[x])
        M = f.base.match(a*x + b)

        if M is None:
            return None

        if f.exp == -1:
            h = log(f.base)
        elif conds != 'piecewise':
            h = f.base**(f.exp + 1) / (f.exp + 1)
        else:
            from sympy.core.relational import Ne
            h1 = log(f.base)
            h2 = f.base**(f.exp + 1) / (f.exp + 1)
            h = Piecewise((h2, Ne(f.exp, -1)), (h1, True))

        return h / M[a]


class RationalFunctionStrategy(IntegrationStrategy):
    """Integrate rational functions using partial fractions"""

    def can_apply(self, f: Expr, x: Expr, **kwargs) -> bool:
        flags = kwargs.get('flags', {})
        return f.is_rational_function(x) and not any(
            flags.get(flag) for flag in ['manual', 'meijerg', 'risch']
        )

    def try_integrate(self, f: Expr, x: Expr, **kwargs) -> Expr | None:
        from .rationaltools import ratint
        try:
            return ratint(f, x)
        except (NotImplementedError, ValueError):
            return None


class TrigStrategy(IntegrationStrategy):
    """Integrate trigonometric functions"""

    def can_apply(self, f: Expr, x: Expr, **kwargs) -> bool:
        flags = kwargs.get('flags', {})
        return not any(flags.get(flag) for flag in ['manual', 'meijerg', 'risch'])

    def try_integrate(self, f: Expr, x: Expr, **kwargs) -> Expr | None:
        from .trigonometry import trigintegrate
        conds = kwargs.get('conds', 'piecewise')
        try:
            return trigintegrate(f, x, conds=conds)
        except (NotImplementedError, ValueError):
            return None


class DeltaStrategy(IntegrationStrategy):
    """Integrate functions with DiracDelta terms"""

    def can_apply(self, f: Expr, x: Expr, **kwargs) -> bool:
        flags = kwargs.get('flags', {})
        return not any(flags.get(flag) for flag in ['manual', 'meijerg', 'risch'])

    def try_integrate(self, f: Expr, x: Expr, **kwargs) -> Expr | None:
        from .deltafunctions import deltaintegrate
        try:
            return deltaintegrate(f, x)
        except (NotImplementedError, ValueError):
            return None


class SingularityStrategy(IntegrationStrategy):
    """Integrate functions with singularity function terms"""

    def can_apply(self, f: Expr, x: Expr, **kwargs) -> bool:
        flags = kwargs.get('flags', {})
        return not any(flags.get(flag) for flag in ['manual', 'meijerg', 'risch'])

    def try_integrate(self, f: Expr, x: Expr, **kwargs) -> Expr | None:
        from .singularityfunctions import singularityintegrate
        try:
            return singularityintegrate(f, x)
        except (NotImplementedError, ValueError):
            return None


class RischStrategy(IntegrationStrategy):
    """Risch algorithm for elementary functions"""

    def can_apply(self, f: Expr, x: Expr, **kwargs) -> bool:
        flags = kwargs.get('flags', {})
        return flags.get('risch') is not False

    def try_integrate(self, f: Expr, x: Expr, **kwargs) -> Expr | None:
        from .risch import risch_integrate
        conds = kwargs.get('conds', 'piecewise')
        try:
            h, i = risch_integrate(f, x, separate_integral=True, conds=conds)
            if i:
                h = h + i.doit(risch=False)
            return h
        except NotImplementedError:
            return None


class HeurischStrategy(IntegrationStrategy):
    """Heuristic Risch algorithm - slower but more comprehensive"""

    def can_apply(self, f: Expr, x: Expr, **kwargs) -> bool:
        flags = kwargs.get('flags', {})
        return flags.get('heurisch') is not False

    def try_integrate(self, f: Expr, x: Expr, **kwargs) -> Expr | None:
        from sympy.integrals.heurisch import heurisch as heurisch_, heurisch_wrapper
        from sympy.polys import PolynomialError
        conds = kwargs.get('conds', 'piecewise')
        try:
            if conds == 'piecewise':
                return heurisch_wrapper(f, x, hints=[])
            else:
                return heurisch_(f, x, hints=[])
        except (PolynomialError, NotImplementedError, ValueError):
            return None


class MeijerGStrategy(IntegrationStrategy):
    """Meijer G-function method for special functions"""

    def can_apply(self, f: Expr, x: Expr, **kwargs) -> bool:
        flags = kwargs.get('flags', {})
        return flags.get('meijerg') is not False

    def try_integrate(self, f: Expr, x: Expr, **kwargs) -> Expr | None:
        from .meijerint import meijerint_indefinite, _debug
        try:
            return meijerint_indefinite(f, x)
        except NotImplementedError:
            _debug('NotImplementedError from meijerint_indefinite')
            return None


class ManualStrategy(IntegrationStrategy):
    """manual integration: mimics hand-solving techniques"""

    def can_apply(self, f: Expr, x: Expr, **kwargs) -> bool:
        flags = kwargs.get('flags', {})
        return flags.get('manual') is not False

    def try_integrate(self, f: Expr, x: Expr, **kwargs) -> Expr | None:
        from sympy.integrals.manualintegrate import manualintegrate
        from sympy.polys import PolynomialError
        from sympy.core.expr import Integral
        conds = kwargs.get('conds', 'piecewise')
        eval_kwargs = kwargs.get('eval_kwargs', {})
        manual = kwargs.get('flags', {}).get('manual')

        try:
            result = manualintegrate(f, x)
            if result is None or isinstance(result, Integral):
                return None

            if result.has(Integral) and not manual:
                # Try to have other algorithms do the integrals manualintegrate
                # can't handle, unless we were asked to use manual only.
                new_eval_kwargs = eval_kwargs.copy()
                new_eval_kwargs['manual'] = False
                new_eval_kwargs['final'] = False
                result = result.func(*[
                    arg.doit(**new_eval_kwargs) if
                    arg.has(Integral) else arg
                    for arg in result.args
                ]).expand(multinomial=False, log=False,
                          power_exp=False, power_base=False)

            if not result.has(Integral):
                return result
            return None
        except (ValueError, PolynomialError):
            return None


# Strategy registry for the main term-by-term loop
# Order matters: strategies are tried in this sequence
TERM_STRATEGIES = [
    ConstantTermStrategy(),
    PowerStrategy(),
    RationalFunctionStrategy(),
    TrigStrategy(),
    DeltaStrategy(),
    SingularityStrategy(),
    RischStrategy(),
    HeurischStrategy(),
    MeijerGStrategy(),
    ManualStrategy(),
]


def try_strategies(
    f: Expr,
    x: Expr,
    strategies: list[IntegrationStrategy],
    **kwargs
) -> Expr | None:
    """
    Try a sequence of integration strategies in order.

    Parameters
    ==========
    f : Expr
        The integrand
    x : Expr
        The integration variable
    strategies : list[IntegrationStrategy]
        List of strategies to try, in order
    **kwargs :
        Context passed to each strategy (eval_kwargs, flags, conds, etc.)

    Returns
    =======
    Expr | None
        The antiderivative if any strategy succeeds, None otherwise
    """
    for strategy in strategies:
        if strategy.can_apply(f, x, **kwargs):
            result = strategy.try_integrate(f, x, **kwargs)
            if result is not None:
                return result
    return None
