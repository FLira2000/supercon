from thermo import nested_formula_parser as parser

def makeup(element):
    print(element)
    print(type(element))

    chem_dict = parser(element)

    return chem_dict

print(makeup('NaCl'))