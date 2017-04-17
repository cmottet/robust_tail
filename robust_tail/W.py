_all__ = ['W']


def W(x, w, rho, limsup, H):
    """Function W as defined in Theorem 5, in the case when nubar = 1

    :param x:
    :param w:
    :param rho: scalars
    :param limsup: scalar
        the limsup of H(x)/x^2 as x goes to infinity
    :param H: function
        function H(x) as defined in program (5) and (EC.19)
    :return: scalar
    """

    if x != w:
        p1 = (rho - w**2)/(rho - 2*w*x + x**2)
        p2 = (w-x)**2/(rho - 2*w*x + x**2)
        x1 = x
        x2 = (rho-w*x)/(w-x)
        output = p1*H(x1) + p2*H(x2)

    if x == w:
        output = H(w) + limsup*(rho - w**2)

    return output
