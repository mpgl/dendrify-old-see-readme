#!/usr/bin/env python
# -*- coding: utf-8 -*-
# conda version : 4.8.3
# conda-build version : 3.18.12
# python version : 3.7.6.final.0
# brian2 version : 2.3 (py37hc9558a2_0)

"""
A class that automatically generates and handles all differential equations
describing a single compartment and the currents (synaptic/dendritic/noise)
passing through it.

---------------------------------------------------------------------------
* ATTRIBUTES (user-defined)

tag: str
    A unique tag used to separate the various compartments belonging to a
    single neuron. It is also used for the creation of compartment-specific
    equations.

model: str
    A keyword for accessing elements of the library. Custom models can also
    be provided but they should be in the same formattable structure as the
    library models. Inexperienced users should avoid using custom models.

neuron: str
    A tag that is used to group compartments of the same neuron.

---------------------------------------------------------------------------
* OTHER ATTRIBUTES (handled by class functions but accessible to user)

_equations: str
    Stores compartment-specific differential equations.

_events: dict
    Stores compartment-specific custom _events.

_event_actions: str
    Executable code that specifies model behaviour during custom _events.

_params: dict
    A helper variable used to store some compartment-specific params

_ephys_object: EphysProperties
    Automatically created when Ephys params are passed to a compartment.
    It also automates the calculations of several other properties such as
    compartment area, absolute capacitance/leakage, etc.

_connections: list
    A list of tuples used for estimating the parameters needed for
    connecting the various compartments. The tuples have the following
    format: ('coupling g name', 'connection type', 'helper objects')

---------------------------------------------------------------------------
* METHODS

- connect
- synapse
- noise
- dspikes
- Na_spikes (called by dspikes)
- Ca_spikes (called by dpsikes)


---------------------------------------------------------------------------
* PROPERTIES

- params
- area
- capacitance
- g_leakage
- g_couples

---------------------------------------------------------------------------
* STATICMETHODS

- g_norm_factor

"""


import sys
import brian2
import numpy as np
from .equations import library
from .ephysproperties import EphysProperties
from brian2.units import ms, pA


class Compartment:

    def __init__(self, tag, model=None, **kwargs):

        self.tag = tag
        self.model = model
        # self.__dict__.update(kwargs)
        self._equations = None
        self._params = None
        self._connections = None
        self._ephys_object = EphysProperties(tag=self.tag, **kwargs)

        # Pick a model template or provide a custom model:
        if self.model in library:
            self._equations = library[self.model].format('_'+self.tag)
        else:
            self._equations = self.model.format('_'+self.tag)

    def connect(self, other, g='half_cylinders'):
        """
        Allows the electrical coupling of two compartments.
        Also assigns a compartment to a neuron.
        Usage -> Comp_1.connect(Comp_2).

        Parameters
        ----------
        other : TYPE
            Description
        g : str, optional
            Description
        """
        # Current from Comp2 -> Comp1
        I_forward = 'I_{1}_{0} = (V_{1}-V_{0}) * g_{1}_{0}  :amp'.format(
            self.tag, other.tag)
        # Current from Comp1 -> Comp2
        I_backward = 'I_{0}_{1} = (V_{0}-V_{1}) * g_{0}_{1}  :amp'.format(
                     self.tag, other.tag)

        # Add them to their respective compartments:
        self._equations += '\n'+I_forward
        other._equations += '\n'+I_backward

        # Include them to the I variable (I_ext -> Inj + new_current):
        self_change = f'= I_ext_{self.tag}'
        other_change = f'= I_ext_{other.tag}'
        self._equations = self._equations.replace(
            self_change, self_change + ' + ' + I_forward.split('=')[0])
        other._equations = other._equations.replace(
            other_change, other_change + ' + ' + I_backward.split('=')[0])

        # add them to connected comps
        if not self._connections:
            self._connections = []
        if not other._connections:
            other._connections = []

        g_to_self = f'g_{other.tag}_{self.tag}'
        g_to_other = f'g_{self.tag}_{other.tag}'

        # when g is specified by user
        if isinstance(g, brian2.units.fundamentalunits.Quantity):
            self._connections.append((g_to_self, 'user', g))
            other._connections.append((g_to_other, 'user', g))

        # when g is a string
        elif isinstance(g, str):
            if g == 'half_cylinders':
                self._connections.append((g_to_self, g, other._ephys_object))
                other._connections.append((g_to_other, g, self._ephys_object))

            elif g.split('_')[0] == "cylinder":
                ctype, tag = g.split('_')
                comp = self if self.tag == tag else other
                self._connections.append(
                    (g_to_self, ctype, comp._ephys_object))
                other._connections.append(
                    (g_to_other, ctype, comp._ephys_object))
        else:
            print('Please select a valid conductance.')

    def synapse(self, channel=None, pre=None, g=None, t_rise=None,
                t_decay=None, scale_g=None, rise_decay=None):
        """
        Adds AMPA/NMDA/GABA synapses from a specified source. The 'source'
        kwarg is used to separate synapses of the same type coming from
        different sources. Usage -> object.add_synapse('type')
        """

        # Make sure that the user provides a synapse source
        if not pre:
            print((f"Warning: <pre> argument missing for '{channel}' "
                   f"synapse @ '{self.tag}'\n"
                   "Program exited.\n"))
            sys.exit()
        # Switch to rise/decay equations if t_rise & t_decated are provided
        if all([t_rise, t_decay]) or rise_decay:
            key = f"{channel}_rd"
        else:
            key = channel

        current_name = f'I_{channel}_{pre}_{self.tag}'

        current_eqs = library[key].format(self.tag, pre)

        to_replace = f'= I_ext_{self.tag}'
        self._equations = self._equations.replace(
            to_replace, f'{to_replace} + {current_name}')
        self._equations += '\n'+current_eqs

        if not self._params:
            self._params = {}

        weight = f"w_{channel}_{pre}_{self.tag}"
        self._params[weight] = 1
        # If user provides a value for g, then add it to _params
        if g:
            self._params[f'g_{channel}_{pre}_{self.tag}'] = g
        if t_rise:
            self._params[f't_{channel}_rise_{pre}_{self.tag}'] = t_rise
        if t_decay:
            self._params[f't_{channel}_decay_{pre}_{self.tag}'] = t_decay
        if scale_g:
            if all([t_rise, t_decay, g]):
                norm_factor = Compartment.g_norm_factor(t_rise, t_decay)
                self._params[f'g_{channel}_{pre}_{self.tag}'] *= norm_factor

    def noise(self, tau=20*ms, sigma=3*pA, mean=0*pA):
        """
        Adds coloured noisy current. More info here:
        https://brian2.readthedocs.io/en/stable/user/models.html#noise
        Usage-> object.noise(optonal kwargs)
        """
        Inoise_name = f'I_noise_{self.tag}'
        noise_eqs = library['noise'].format(self.tag)
        to_change = f'= I_ext_{self.tag}'
        self._equations = self._equations.replace(to_change,
                                                  f'{to_change} + {Inoise_name}')
        self._equations += '\n'+noise_eqs

        # Add _params:
        if not self._params:
            self._params = {}
        self._params[f'tau_noise_{self.tag}'] = tau
        self._params[f'sigma_noise_{self.tag}'] = sigma
        self._params[f'mean_noise_{self.tag}'] = mean

    @property
    def parameters(self):
        d_out = {}
        for i in [self._params, self.g_couples]:
            if i:
                d_out.update(i)
        if self._ephys_object:
            d_out.update(self._ephys_object.parameters)
        return d_out

    @property
    def area(self):
        try:
            return self._ephys_object.area
        except AttributeError:
            print(("Error: Missing Parameters\n"
                   f"Cannot calculate the area of <{self.tag}>, "
                   "returned None instead.\n"))

    @property
    def capacitance(self):
        try:
            return self._ephys_object.capacitance
        except AttributeError:
            print(("Error: Missing Parameters\n"
                   f"Cannot calculate the capacitance of <{self.tag}>, "
                   "returned None instead.\n"))

    @property
    def g_leakage(self):
        try:
            return self._ephys_object.g_leakage
        except AttributeError:
            print(("Error: Missing Parameters\n"
                   f"Cannot calculate the g leakage of <{self.tag}>, "
                   "returned None instead.\n"))

    @property
    def equations(self):
        return self._equations

    @property
    def g_couples(self):
        # If not _connections have been specified yet
        if not self._connections:
            return None

        d_out = {}
        for i in self._connections:
            # If ephys objects are not created yet
            if not i[2]:
                return None

            name, ctype, helper_ephys = i[0], i[1], i[2]

            if ctype == 'half_cylinders':
                value = EphysProperties.g_couple(
                    self._ephys_object, helper_ephys)

            elif ctype == 'cylinder':
                value = helper_ephys.g_cylinder

            elif ctype == 'user':
                value = helper_ephys

            d_out[name] = value
        return d_out

    @staticmethod
    def g_norm_factor(trise, tdecay):
        tpeak = (tdecay*trise / (tdecay-trise)) * np.log(tdecay/trise)
        factor = (((tdecay*trise) / (tdecay-trise))
                  * (-np.exp(-tpeak/trise) + np.exp(-tpeak/tdecay))
                  / ms)
        return 1/factor


class Soma(Compartment):
    """docstring for Soma
    """

    def __init__(self, tag, model=None, **kwargs):
        super().__init__(tag, model, **kwargs)
        self._events = None

    def __str__(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        ephys_dict = self._ephys_object.__dict__
        ephys = '\n'.join([f"\u2192 {i}:\n  [{ephys_dict[i]}]\n"
                           for i in ephys_dict])
        equations = self.equations.replace('\n', '\n   ')

        parameters = '\n'.join([f"   '{i[0]}': {i[1]}"
                                for i in self.parameters.items()
                                ]) if self.parameters else '   None'

        msg = (f"OBJECT TYPE:\n\n  {self.__class__}\n\n"
               f"{'-'*45}\n\n"
               f"USER PARAMETERS:\n\n{ephys}"
               f"\n{'-'*45}\n\n"
                "PROPERTIES: \n\n"
               f"\u2192 equations:\n   {equations}\n\n"
               f"\u2192 parameters:\n{parameters}\n")
        return msg


class Dendrite(Compartment):
    """docstring for Soma"""

    def __init__(self, tag, model='passive', **kwargs):
        super().__init__(tag, model, **kwargs)
        self._events = None
        self._event_actions = None

    def __str__(self):
        """Summary

        Returns
        -------
        TYPE
            Description
        """
        ephys_dict = self._ephys_object.__dict__
        
        ephys = '\n'.join([f"\u2192 {i}:\n    [{ephys_dict[i]}]\n"
                           for i in ephys_dict])

        equations = self.equations.replace('\n', '\n    ')

        events = '\n'.join([f"    '{key}': '{self.events[key]}'"
                            for key in self.events
                            ]) if self.events else '    None'

        parameters = '\n'.join([f"    '{i[0]}': {i[1]}"
                                for i in self.parameters.items()
                                ]) if self.parameters else '    None'

        msg = (f"OBJECT TYPE:\n\n  {self.__class__}\n\n"
               f"{'-'*45}\n\n"
               f"USER PARAMETERS:\n\n{ephys}"
               f"\n{'-'*45}\n\n"
                "PROPERTIES: \n\n"
               f"\u2192 equations:\n    {equations}\n\n"
               f"\u2192 events:\n{events}\n\n"
               f"\u2192 parameters:\n{parameters}\n")
        return msg

    def dspikes(self, channel, threshold=None, g_rise=None, g_fall=None):
        if channel == 'Na':
            self.Na_spikes(threshold=threshold, g_rise=g_rise, g_fall=g_fall)
        elif channel == 'Ca':
            self.Ca_spikes(threshold=threshold, g_rise=g_rise, g_fall=g_fall)

    def Na_spikes(self, threshold=None, g_rise=None, g_fall=None):
        """
        Adds Na spike currents (rise->I_Na, decay->I_Kn) and  other variables
        for controlling custom _events.
        Usage-> object.Na_spikes()
        """
        # The following code creates all necessary equations for dspikes:
        tag = self.tag
        dspike_currents = f'I_Na_{tag} + I_Kn_{tag}'

        I_Na_eqs = f'I_Na_{tag} = g_Na_{tag} * (E_Na-V_{tag})  :amp'
        I_Kn_eqs = f'I_Kn_{tag} = g_Kn_{tag} * (E_K-V_{tag})  :amp'

        g_Na_eqs = f'dg_Na_{tag}/dt = -g_Na_{tag}/tau_Na  :siemens'
        g_Kn_eqs = f'dg_Kn_{tag}/dt = -g_Kn_{tag}/tau_Kn  :siemens'

        I_Na_check = f'allow_I_Na_{tag}  :boolean'
        I_Kn_check = f'allow_I_Kn_{tag}  :boolean'
        refractory_var = f'timer_Na_{tag}  :second'
        to_replace = f'= I_ext_{tag}'
        self._equations = self._equations.replace(
            to_replace, f'{to_replace} + {dspike_currents}')
        self._equations += '\n'.join(['', I_Na_eqs, I_Kn_eqs, g_Na_eqs, g_Kn_eqs,
                                      I_Na_check, I_Kn_check, refractory_var])

        # Create all necessary custom _events for dspikes:
        condition_I_Na = library['condition_I_Na']
        condition_I_Kn = library['condition_I_Kn']
        if not self._events:
            self._events = {}
        self._events[f"activate_I_Na_{tag}"] = condition_I_Na.format(tag)
        self._events[f"activate_I_Kn_{tag}"] = condition_I_Kn.format(tag)

        # Specify what is going to happen inside run_on_event()
        if not self._event_actions:
            self._event_actions = library['run_on_Na_spike'].format(tag)
        else:
            self._event_actions += "\n" + library['run_on_Na_spike'].format(tag)
        # Include params needed
        if not self._params:
            self._params = {}
        if threshold:
            self._params[f"Vth_Na_{self.tag}"] = threshold
        if g_rise:
            self._params[f"g_Na_{self.tag}_max"] = g_rise
        if g_fall:
            self._params[f"g_Kn_{self.tag}_max"] = g_fall

    def Ca_spikes(self, threshold=None, g_rise=None, g_fall=None):
        """
        Adds Na spike currents and some other variables
        for controlling custom _events.
        Usage-> object.Ca_spikes()
        """
        # The following code creates all necessary equations for dspikes:
        tag = self.tag
        dspike_currents = f'I_Ca_{tag} + I_Kc_{tag}'
        I_Ca_eqs = f'dI_Ca_{tag}/dt = -I_Ca_{tag}/tau_Ca  :amp'
        I_Kc_eqs = f'dI_Kc_{tag}/dt = -I_Kc_{tag}/tau_Kc  :amp'
        I_Ca_check = f'allow_I_Ca_{tag}  :boolean'
        I_Kc_check = f'allow_I_Kc_{tag}  :boolean'
        refractory_var = f'timer_Ca_{tag}  :second'
        to_replace = f'= I_ext_{tag}'
        self._equations = self._equations.replace(
            to_replace, f'{to_replace} + {dspike_currents}')
        self._equations += '\n'.join(['', I_Ca_eqs, I_Kc_eqs, I_Ca_check,
                                      I_Kc_check, refractory_var])

        # Create all necessary custom _events for dspikes:
        condition_I_Ca = library['condition_I_Ca']
        condition_I_Kc = library['condition_I_Kc']
        if not self._events:
            self._events = {}
        self._events[f"activate_I_Ca_{tag}"] = condition_I_Ca.format(tag)
        self._events[f"activate_I_Kc_{tag}"] = condition_I_Kc.format(tag)

        # Specify what is going to happen inside run_on_event()
        if not self._event_actions:
            self._event_actions = library['run_on_Ca_spike'].format(tag)
        else:
            self._event_actions += "\n" + library['run_on_Ca_spike'].format(tag)
        # Include params needed
        if not self._params:
            self._params = {}
        if threshold:
            self._params[f"Vth_Ca_{self.tag}"] = threshold
        if g_rise:
            self._params[f"g_Ca_{self.tag}_max"] = g_rise
        if g_fall:
            self._params[f"g_Kc_{self.tag}_max"] = g_fall

    @property
    def events(self):
        return self._events

    @property
    def event_actions(self):
        return self._event_actions
