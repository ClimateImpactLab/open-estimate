from . import formatting
from .formatting import FormatElement

def call(func, description=None, *args):
    """Return a representation of this call.  Any elements in args can
    be given their own FormatElements in the final dictionary.
    """
    
    if len(args) == 0:
        funcvar = formatting.get_function()
        return {'main': FormatElement(funcvar + "()", [funcvar + '()'], is_primitive=True),
                funcvar + '()': FormatElement(description, is_abstract=True)}

    julia = julia_function(func, *tuple(map(formatting.get_repstr, args)))
    if julia:
        elements = {'main': FormatElement(julia)}
        for arg in args:
            if isinstance(arg, FormatElement):
                elements['main'].dependencies.extend(arg.dependencies)
        return elements
    
    if len(args) == 1:
        funcvar = formatting.get_function()

        argvar = formatting.call_argvar(args[0])

        if isinstance(argvar, FormatElement):
            return {'main': FormatElement("%s(%s)" % (funcvar, argvar.repstr),
                                          [funcvar + "(x)"] + argvar.dependencies, is_primitive=True),
                    funcvar + "(x)": FormatElement(description, args, is_abstract=True)}
        elif isinstance(args[0], FormatElement):
            return {'main': FormatElement("%s(%s)" % (funcvar, argvar),
                                          [funcvar + "(x)", argvar], is_primitive=True),
                    funcvar + "(x)": FormatElement(description, args, is_abstract=True),
                    argvar: args[0]}
        else:
            return {'main': FormatElement("%s(%s)" % (funcvar, args[0]),
                                          [funcvar + "(x)"], is_primitive=True),
                    funcvar + "(x)": FormatElement(description, args, is_abstract=True)}
            
    elif len(args) == 2:
        funcvar = formatting.get_function()

        argname0 = formatting.call_argvar(args[0])
        argname1 = formatting.call_argvar(args[1])
            
        result = {'main': FormatElement("%s(%s, %s)" % (funcvar, argname0, argname1),
                                        [funcvar + "(x, y)"], is_primitive=True),
                  funcvar + "(x, y)": FormatElement(description, is_abstract=True)}
        if isinstance(args[0], FormatElement):
            result[argname0] = args[0]
            result['main'].dependencies.append(argname0)
        if isinstance(args[1], FormatElement):
            result[argname1] = args[1]
            result['main'].dependencies.append(argname1)

        return result

    raise RuntimeError("Cannot format %s(%s)" % (func, ', '.join(["%s" % arg for arg in args])))
            
def julia_function(func, *args):
    if len(args) == 1:
        interp = formatting.interpret1(func)
        if interp == 'identity':
            return args[0]

    if len(args) == 2:
        interp = formatting.interpret2(func)
        if interp == '/':
            return "(%s) / (%s)" % args
        elif interp == '-':
            return "%s - (%s)" % args
        elif interp == '*':
            return "(%s) * (%s)" % args
