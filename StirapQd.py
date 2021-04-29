import scipy
import time_evolving_mpo as tempo
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from progress.bar import Bar


# create a class for the system
class StirapQd:
    """
            A STIRAP system with default parameters for a QD
            """

    def __init__(self, process_tensor=tempo.import_process_tensor('data/process_tensor_default')):
        """
                Initialise the system
                """
        self.process_tensor = process_tensor
        self.states = []
        self.final_states = []

    def compute_process_tensor(self,
                               alpha=0.126,
                               omega_cutoff=3.04,
                               time_max=20.0,
                               max_correlation_time=2.0,
                               dt=0.1,
                               dkmax=20,
                               epsrel=(10 ** (-5))
                               ):
        """
                    Compute new process tensor (Default QD parameters used)

                    Parameters
                    ---
                    alpha : float
                        alpha for correlations
                    omega_cutoff : float
                        cut off for the correlations
                    time_max : float
                        time of dynamics computation
                    max_correlation_time : float
                        max correlation time
                    dt : float
                        time step
                    dkmax : int
                        dkmax for TEMPO
                    epsrel : float
                        epsrel for TEMPO

                    Return
                    ---

                    """

        # define spectral density
        def j(w):
            return 2 * alpha * (w ** 3) * (omega_cutoff ** -2)

        # setup correlations
        correlations = tempo.correlations.CustomSD(j_function=j,
                                                   cutoff=omega_cutoff,
                                                   cutoff_type='gaussian',
                                                   max_correlation_time=max_correlation_time)
        # setup bath
        bath = tempo.Bath(np.array(
            [[1 + 0.j, 0. + 0.j, 0. + 0.j],
             [0. + 0.j, -1 + 0.j, 0. + 0.j],
             [0. + 0.j, 0. + 0.j, 1 + 0.j]]),
            correlations)
        # setup parameters
        parameters = tempo.PtTempoParameters(dt=dt,
                                             dkmax=dkmax,
                                             epsrel=epsrel,
                                             name="Parameters")
        # compute the process tensor
        process_tensor = tempo.pt_tempo_compute(bath=bath,
                                                parameters=parameters,
                                                start_time=0.0,
                                                end_time=time_max,
                                                description_dict={
                                                    'alpha': alpha,
                                                    'omega_cutoff': omega_cutoff,
                                                    'time_max': time_max,
                                                    'max_correlation_time': max_correlation_time,
                                                    'dt': dt,
                                                    'dkmax': dkmax,
                                                    'epsrel': epsrel
                                                }
                                                )

        self.process_tensor = process_tensor

    def compute(self,
                final=False,
                parameters='empty',
                amplitude=1.0,
                delay=1.5,
                width=1.5,
                delta=0.0,
                markovian=False,
                alpha=0.126,
                old=False
                ):
        """
                Compute states at all times

                Parameters
                ---
                final : boolean
                    Whether to only get final state, or calculate all of them
                parameters : List
                    List of parameters in form [[name,start,step size,steps],[...]...]
                amplitude : float
                    Amplitude of beam in meV
                delay : float
                    Delay between the two pulses, counter intuitive ordering
                width : float
                    Width of the beams
                delta : float
                    Detuning of the pulses
                markovian : boolean
                    Whether to include Linblad operators
                alpha : float
                    Strength of coupling
                old : boolean
                    whether to use old linblad or new
                Return
                ---

                """

        # create dictionary to convert between the input parameters and indexes
        index_dic = {
            'amplitude': 0,
            'delay': 1,
            'width': 2,
            'delta': 3
        }

        # load data from the process tensor
        info_dict = self.process_tensor.description_dict['pt_tempo_description_dict']

        omega_cutoff = info_dict['omega_cutoff']
        time_max = info_dict['time_max']
        max_correlation_time = info_dict['max_correlation_time']
        dt = info_dict['dt']
        dkmax = info_dict['dkmax']
        epsrel = info_dict['epsrel']

        # check to see if parameters are empty and set to default if it is
        if parameters == 'empty':
            parameters = [['amplitude', amplitude, 1.0, 1]]

        # get number of parameters and setup the index array
        num_parameters = len(parameters)
        index_array = np.repeat(0, num_parameters)

        # convert the name into a readable index
        for i in range(num_parameters):
            parameters[i][0] = index_dic[parameters[i][0]]
        # parameters[:, 0] = np.array([index_dic[parameter] for parameter in parameters[:, 0]])
        # convert parameters into 4 1d arrays for easier use
        parameters_swap = np.swapaxes(parameters, 0, 1)
        choices = np.repeat(0, num_parameters)
        names = np.choose(choices, parameters_swap)

        choices = np.repeat(1, num_parameters)
        starts = np.choose(choices, parameters_swap)

        choices = np.repeat(2, num_parameters)
        step_sizes = np.choose(choices, parameters_swap)

        choices = np.repeat(3, num_parameters)
        steps = np.choose(choices, parameters_swap)

        steps = steps.astype(int)
        # create array to store the states
        states_array = np.zeros(tuple(steps), dtype=list)

        # setup starting values of parameters
        parameter_values = [amplitude, delay, width, delta]

        # setup gaussian pulse
        def gaussian(t, amplitude_i, delay_i, width_i):
            return amplitude_i * np.exp(-((t - (time_max / 2) - delay_i) ** 2) / (2 * (width_i ** 2)))

        # iterate over the indexes, calculating the value at each point
        done = False
        with Bar('Computing %(elapsed_td)ss',
                 max=steps.prod(),
                 suffix='%(percent)d%% eta:%(eta_td)ss') as bar:
            while not done:

                # set values of parameters
                for i in range(num_parameters):
                    parameter_values[int(names[i])] = starts[i] + index_array[i] * step_sizes[i]

                # setup pulses and hamiltonian
                def omega_p(t):
                    return gaussian(t,
                                    parameter_values[index_dic['amplitude']] * 1.519,
                                    parameter_values[index_dic['delay']] / 2,
                                    parameter_values[index_dic['width']])

                def omega_s(t):
                    return gaussian(t,
                                    parameter_values[index_dic['amplitude']] * 1.519,
                                    -parameter_values[index_dic['delay']] / 2,
                                    parameter_values[index_dic['width']])

                def hamiltonian(t):
                    return 0.5 * np.array(
                        [[0. + 0.j, omega_p(t) + 0.j, 0. + 0.j],
                         [omega_p(t) + 0.j, 2 * parameter_values[index_dic['delta']] * 1.519 + 0.j, omega_s(t) + 0.j],
                         [0. + 0.j, omega_s(t) + 0.j, 0. + 0.j]])

                # define spectral density
                def j(w):
                    return 2 * alpha * (w ** 3) * (omega_cutoff ** -2) * np.exp(-(w ** 2) / (omega_cutoff ** 2))

                # define root mean square
                def omega_rms(t):
                    return np.sqrt(omega_p(t) ** 2 + omega_s(t) ** 2)

                # setup the system
                if markovian:
                    if not(old):
                        def L(t):
                            omega_pt = omega_p(t)
                            omega_st = omega_s(t)
                            omega_rmst = omega_rms(t)
                            # setting numerical limit, to check how the importance of these low frequency parts
                            if omega_rmst < 10 ** -5 * parameter_values[index_dic['amplitude']] * 1.519:
                                return np.zeros(9).reshape(3, 3)
                            else:
                                return 0.5 * np.array([[(omega_pt ** 2) / (omega_rmst ** 2) + 0.j,
                                                        omega_pt / omega_rmst + 0.j,
                                                        (omega_pt * omega_st) / (omega_rmst ** 2) + 0.j],
                                                       [-omega_pt / omega_rmst + 0.j,
                                                        -1. + 0.j,
                                                        -omega_st / omega_rmst + 0.j],
                                                       [(omega_pt * omega_st) / (omega_rmst ** 2) + 0.j,
                                                        omega_st / omega_rmst + 0.j,
                                                        (omega_st ** 2) / (omega_rmst ** 2) + 0.j]])

                        def gamma(t):
                            return 2 * np.pi * j(omega_rms(t))
                    else:
                        def L(t):
                            return np.array([[1.0 + 0.j,
                                              0.0 + 0.j,
                                              0.0 + 0.j],
                                             [0.0 + 0.j,
                                              -1.0 + 0.j,
                                              0.0 + 0.j],
                                             [0.0 + 0.j,
                                              0.0 + 0.j,
                                              1.0 + 0.j]])

                        def gamma(t):
                            return alpha

                    system = tempo.TimeDependentSystem(hamiltonian, lindblad_operators=[L],
                                                       gammas=[gamma])
                else:
                    system = tempo.TimeDependentSystem(hamiltonian)

                # compute amd save states
                if final:
                    states_array[tuple(index_array)] = \
                        self.process_tensor.compute_final_state_from_system(system=system,
                                                                            initial_state=np.array([
                                                                                [1.0 + 0.j, 0. + 0.j, 0. + 0.j],
                                                                                [0. + 0.j, 0. + 0.j, 0. + 0.j],
                                                                                [0. + 0.j, 0. + 0.j, 0. + 0.j]]))
                else:
                    dynamics = self.process_tensor.compute_dynamics_from_system(system=system,
                                                                                initial_state=np.array([
                                                                                    [1.0 + 0.j, 0. + 0.j, 0. + 0.j],
                                                                                    [0. + 0.j, 0. + 0.j, 0. + 0.j],
                                                                                    [0. + 0.j, 0. + 0.j, 0. + 0.j]]))
                    states_array[tuple(index_array)] = dynamics.states

                # increment the index
                index_array[0] += 1
                for i in range(num_parameters):
                    if index_array[i] == steps[i] and i != num_parameters - 1:
                        index_array[i + 1] += 1
                        index_array[i] = 0
                    elif index_array[i] == steps[i]:
                        done = True

                bar.next()

        # convert the index back into a readable name
        reversed_index_dic = {value: key for (key, value) in index_dic.items()}
        for i in range(num_parameters):
            parameters[i][0] = reversed_index_dic[parameters[i][0]]

        # store all the info in a dictionary along side the states
        info = {'alpha': alpha,
                'omega_cutoff': omega_cutoff,
                'time_max': time_max,
                'max_correlation_time': max_correlation_time,
                'dt': dt,
                'dkmax': dkmax,
                'epsrel': epsrel,
                'amplitude': amplitude,
                'width': width,
                'delay': delay,
                'delta': delta,
                'parameters': parameters,
                'index dict': index_dic
                }
        # set the self variable states to the computed states
        if final:
            self.final_states = [states_array, info]
        else:
            self.states = [states_array, info]

    def import_process_tensor(self,
                              path):
        """
                Import process tensor into the object

                Parameters
                ---
                path : string
                    file path

                Return
                ---

                """
        if type(path) == type('string'):
            process_tensor = tempo.import_process_tensor(path)
            self.process_tensor = process_tensor
        else:
            print("Please insert a valid path")

    def import_states(self,
                      path):
        """
                Import states into the object

                Parameters
                ---
                path : string
                    file path

                Return
                ---

                """
        if type(path) == type('string'):
            states = np.load(path, allow_pickle=True)
            self.states = states
        else:
            print("Please insert a valid path")

    def import_final_states(self,
                            path):
        """
                Import final states into the object

                Parameters
                ---
                path : string
                    file path

                Return
                ---

                """
        if type(path) == type('string'):
            final_states = np.load(path, allow_pickle=True)
            self.final_states = final_states
        else:
            print("Please insert a valid path")

    def export_process_tensor(self,
                              path,
                              overwrite=False):
        """
             Export process tensor to file path

             Parameters
                ---
                path : string
                    file path
                overwrite : boolean
                    whether to overwrite the filepath

                Return
                ---


                """
        if type(path) == type('string'):
            self.process_tensor.export(path, overwrite=overwrite)
        else:
            print("Please insert a valid path")

    def export_states(self,
                      path):
        """
                Export states to file path

                Parameters
                ---
                path : string
                    file path

                Return
                ---

                """
        if type(path) == type('string'):
            np.save(path, self.states)
        else:
            print("Please insert a valid path")

    def export_final_states(self,
                            path):
        """
                Export final states to file path

                Parameters
                ---
                path : string
                    file path

                Return
                ---

                """
        if type(path) == type('string'):
            np.save(path, self.final_states)
        else:
            print("Please insert a valid path")

    def convert_to_final(self):
        """
        Converts the states at all times, to only the final states, for use in comparisons
        """

        states = self.states[0]
        info = self.states[1]
        parameters = info['parameters']
        steps = []
        for parameter in parameters:
            steps.append(parameter[3])

        # get number of parameters and setup the index array
        num_parameters = len(parameters)
        index_array = np.repeat(0, num_parameters)

        final_states = np.zeros(tuple(steps), dtype=list)

        done = False
        while not done:
            state = states[tuple(index_array)]

            final_states[tuple(index_array)] = state[-1]

            # increment the index
            index_array[0] += 1
            for i in range(num_parameters):
                if index_array[i] == steps[i] and i != num_parameters - 1:
                    index_array[i + 1] += 1
                    index_array[i] = 0
                elif index_array[i] == steps[i]:
                    done = True

        self.final_states = [final_states, info]

    def general_plots(self,
                      amplitude=1.0,
                      delay=1.0,
                      width=1.0,
                      delta=0.0
                      ):
        """
                Create plots from states with values closet to specified parameters

                Parameters
                ---
                amplitude : float
                    amplitude of beams
                delay : float
                    delay of beams
                width : float
                    width of beams
                delta : float
                    detuning of beams

                Return
                ---

                """

        # create a dictionary to make sure everything is indexed properly
        parameters_dic = {
            'amplitude': amplitude,
            'delay': delay,
            'width': width,
            'delta': delta
        }

        # get the dictionary
        info_dic = self.states[1]

        # get time parameters
        time_max = info_dic['time_max']  # time of dynamics computation
        dt_0 = info_dic['dt']  # time step of the dynamics

        # extract which parameters were used
        parameters = info_dic['parameters']
        parameters_len = len(parameters)

        # get steps
        steps = np.zeros(parameters_len, dtype=int)
        for i in range(parameters_len):
            steps[i] = parameters[i][3]

        # find the indexes
        index_array = np.repeat(0, parameters_len)
        for i in range(parameters_len):
            index_array[i] = int(round((parameters_dic[parameters[i][0]] - parameters[i][1]) / parameters[i][2]))
            if index_array[i] >= steps[i]:
                index_array[i] = steps[i] - 1
            elif index_array[i] < 0:
                index_array[i] = 0

        # get the states for this index
        states = self.states[0][tuple(index_array)]

        # set the values for this index
        for i in range(parameters_len):
            info_dic[parameters[i][0]] = parameters[i][1] + index_array[i] * parameters[i][2]

        amplitude = info_dic['amplitude']
        delay = info_dic['delay']
        width = info_dic['width']
        delta = info_dic['delta']

        # create time array
        time_array = np.arange(0.0, time_max + dt_0, dt_0)

        # diagonal elements
        p11_array = []
        p22_array = []
        p33_array = []

        # 1-2 correlations
        p12_array = []
        p21_array = []

        # 1-3 correlations
        p13_array = []
        p31_array = []

        # 2-3 correlations
        p23_array = []
        p32_array = []

        # purity
        purity_array = []

        # get the values
        for i in range(int(time_max / dt_0) + 1):
            # diagonal elements
            p11_array.append(np.abs(states[i][0][0]))
            p22_array.append(np.abs(states[i][1][1]))
            p33_array.append(np.abs(states[i][2][2]))

            # 1-2 correlations
            p12_array.append(np.abs(states[i][0][1]))
            p21_array.append(np.abs(states[i][1][0]))

            # 1-3 correlations
            p13_array.append(np.abs(states[i][0][2]))
            p31_array.append(np.abs(states[i][2][0]))

            # 2-3 correlations
            p23_array.append(np.abs(states[i][1][2]))
            p32_array.append(np.abs(states[i][2][1]))

            # get purity
            purity_array.append(np.abs(np.trace(states[i] @ states[i])))

        # setup gaussian pulse
        def gaussian(t, amplitude_i, delay_i, width_i):
            return amplitude_i * np.exp(-((t - (time_max / 2) - delay_i) ** 2) / (2 * (width_i ** 2)))

        # define the pulses
        def omega_p(t):
            return gaussian(t, amplitude, delay / 2, width)

        def omega_s(t):
            return gaussian(t, amplitude, -delay / 2, width)

        # create the plot figure
        fig = plt.figure(figsize=[10.0, 10.0])
        title_font = {'size': 20}
        font = {'size': 15}

        # creat the title
        mpl.rc('font', **title_font)
        fig.suptitle('General plots')
        mpl.rc('font', **font)

        # plot populations
        ax = fig.add_subplot(411)
        ax.plot(time_array, p11_array, 'b', label='level 1')
        ax.plot(time_array, p22_array, 'r', label='level 2')
        ax.plot(time_array, p33_array, 'y', label='level 3')
        ax.legend()
        plt.grid(True, axis='x')
        ax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        ax.set_title('delay = ' + str(round(delay, 3)) + '(ps), width = ' +
                     str(round(width, 3)) + '(ps), amplitude = ' +
                     str(round(amplitude, 3)) + '(meV)  delta = ' +
                     str(round(delta, 3)) + '(meV)')
        ax.set_ylabel('Population')


        # plot correlations
        ax2 = fig.add_subplot(412)
        ax2.plot(time_array, p12_array, 'purple', label='1-2')
        ax2.plot(time_array, p13_array, 'g', label='1-3')
        ax2.plot(time_array, p23_array, 'orange', label='2-3')
        ax2.legend()
        plt.grid(True, axis='x')
        ax2.set_ylim([0, 1])
        ax2.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        ax2.set_ylabel('Abs Coefficient')


        # add in the beams
        ax3 = fig.add_subplot(413)
        ax3.plot(time_array, omega_p(time_array), 'purple', label='Pump')
        ax3.plot(time_array, omega_s(time_array), 'orange', label='Stokes')
        ax3.legend()
        plt.grid(True, axis='x')

        ax3.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        ax3.set_ylabel('Amplitude (meV)')


        ax4 = fig.add_subplot(414)

        ax4.plot(time_array, purity_array)
        ax4.set_xlabel('Time (ps)')
        ax4.set_ylabel('Purity')
        ax4.set_ylim([0, 1.1])
        # show plot
        plt.grid(True, axis='x')


        plt.show()


    def compare_1d(self,
                   parameter_x,
                   amplitude=1.0,
                   delay=1.0,
                   width=1.0,
                   delta=0.0
                   ):
        """
        create plots for varying 1 parameter

        Parameters
        ---
        parameter_x : string
            parameter to be varied over data set
        amplitude : float
            amplitude of beams
        delay : float
            delay of beams
        width : float
            width of beams
        delta : float
            detuning of beams

        """
        # check parameter is a string
        if type(parameter_x) == type('string'):

            # get the dictionary
            info_dic = self.final_states[1]

            # extract which parameters were used
            parameters = info_dic['parameters']
            parameters_len = len(parameters)

            # find the index
            parameter_x_index = -1
            for i in range(parameters_len):
                if parameters[i][0] == parameter_x.lower():
                    parameter_x_index = i
                    break
            # check that the parameter is valid
            if parameter_x_index >= 0:
                # create a dictionary to make sure everything is indexed properly
                parameters_dic = {
                    'amplitude': amplitude,
                    'delay': delay,
                    'width': width,
                    'delta': delta
                }

                # get steps
                steps = np.zeros(parameters_len, dtype=int)
                for i in range(parameters_len):
                    steps[i] = parameters[i][3]

                # find the base indexes for set values
                base_indexes = np.zeros(parameters_len, dtype=int)
                for i in range(parameters_len):
                    base_indexes[i] = int(
                        round((parameters_dic[parameters[i][0]] - parameters[i][1]) / parameters[i][2]))
                    # truncate to allowed indexes
                    if base_indexes[i] >= steps[i]:
                        base_indexes[i] = steps[i] - 1
                    elif base_indexes[i] < 0:
                        base_indexes[i] = 0

                # create the data iterating over given parameter
                final_states = self.final_states[0]
                index_array = base_indexes
                plot_states = []
                for i in range(steps[parameter_x_index]):
                    index_array.put(parameter_x_index, i)
                    plot_states.append(np.real(final_states[tuple(index_array)][2][2]))

                # make array of parameter_x values
                parameter_x_array = np.zeros(steps[parameter_x_index])
                for i in range(steps[parameter_x_index]):
                    parameter_x_array[i] = parameters[parameter_x_index][1] + i * parameters[parameter_x_index][2]

                # set the values for this index
                for i in range(parameters_len):
                    info_dic[parameters[i][0]] = parameters[i][1] + base_indexes[i] * parameters[i][2]

                amplitude = info_dic['amplitude']
                delay = info_dic['delay']
                width = info_dic['width']
                delta = info_dic['delta']

                # create the labels
                label_dict = {
                    'amplitude': 'amplitude = ' + str(round(amplitude, 3)) + 'meV ',
                    'delay': 'delay = ' + str(round(delay, 3)) + 'ps ',
                    'width': 'width = ' + str(round(width, 3)) + 'ps ',
                    'delta': 'delta = ' + str(round(delta, 3)) + 'meV '
                }

                # remove the parameter_x label
                del label_dict[parameter_x]

                # creat the whole label
                values = label_dict.values()
                label = ', '.join(values)

                fig = plt.figure(figsize=[10.0, 10.0])
                title_font = {'size': 30}
                font = {'size': 15}

                mpl.rc('font', **title_font)
                fig.suptitle('Fidelity changing ' + parameter_x)
                mpl.rc('font', **font)
                ax = fig.add_subplot(111)
                # make plot and labels
                ax.plot(parameter_x_array, plot_states)
                ax.set_title(label)

                ax.set_xlabel(parameter_x)
                ax.set_ylabel('Fidelity')
                plt.show()

            else:
                print('please enter a valid parameter')
        else:
            print('please enter parameter name as a string')

    def compare_2d(self,
                   parameter_x,
                   parameter_y,
                   amplitude=1.0,
                   delay=1.0,
                   width=1.0,
                   delta=0.0,
                   infidelity=False,
                   truncate=False,
                   truncate_value=0.95,
                   surface=False
                   ):
        """
        create plots for varying 2 parameters

        Parameters
        ---
        parameter_x : string
            parameter to be varied over data set along x axis
        parameter_y : string
            parameter to be varied over data set along y axis
        amplitude : float
            amplitude of beams
        delay : float
            delay of beams
        width : float
            width of beams
        delta : float
            detuning of beams
        infidelity : boolean
            plot the infidelity or the fidelity
        truncate : boolean
            if plotting infertility, then truncate all values below 0.95, and rescale
        truncate_value : float
            value to truncate at
        surface : boolean
            whether to plot the 3D surface

        """
        # check parameter is a string
        if type(parameter_x) == type('string') and type(parameter_y) == type('string'):

            # get the dictionary
            info_dic = self.final_states[1]

            # extract which parameters were used
            parameters = info_dic['parameters']
            parameters_len = len(parameters)

            # find the indexes
            parameter_x_index = -1
            parameter_y_index = -1
            for i in range(parameters_len):
                if parameters[i][0] == parameter_x.lower():
                    parameter_x_index = i
                elif parameters[i][0] == parameter_y.lower():
                    parameter_y_index = i
            # check that the parameters are valid
            if parameter_x_index >= 0 and parameter_y_index >= 0:
                # create a dictionary to make sure everything is indexed properly
                parameters_dic = {
                    'amplitude': amplitude,
                    'delay': delay,
                    'width': width,
                    'delta': delta
                }

                # get steps
                steps = np.zeros(parameters_len, dtype=int)
                for i in range(parameters_len):
                    steps[i] = parameters[i][3]

                # find the base indexes for set values
                base_indexes = np.zeros(parameters_len, dtype=int)
                for i in range(parameters_len):
                    base_indexes[i] = int(
                        round((parameters_dic[parameters[i][0]] - parameters[i][1]) / parameters[i][2]))
                    # truncate to allowed indexes
                    if base_indexes[i] >= steps[i]:
                        base_indexes[i] = steps[i] - 1
                    elif base_indexes[i] < 0:
                        base_indexes[i] = 0

                # create the data iterating over given parameters
                final_states = self.final_states[0]
                index_array = base_indexes
                plot_states = np.zeros((steps[parameter_y_index], steps[parameter_x_index]))
                for x in range(steps[parameter_x_index]):
                    index_array.put(parameter_x_index, x)
                    for y in range(steps[parameter_y_index]):
                        index_array.put(parameter_y_index, y)
                        final_state = final_states[tuple(index_array)]

                        if infidelity:
                            plot_states[y, x] = -np.log(1 - np.real(final_state[2][2]))
                        elif truncate:
                            plot_states[y, x] = np.real(final_state[2][2]) - truncate_value
                        else:
                            plot_states[y, x] = np.real(final_state[2][2])

                # make array of parameters values
                parameter_x_array = np.zeros(steps[parameter_x_index])
                parameter_y_array = np.zeros(steps[parameter_y_index])
                for i in range(steps[parameter_x_index]):
                    parameter_x_array[i] = parameters[parameter_x_index][1] + i * parameters[parameter_x_index][2]
                for i in range(steps[parameter_y_index]):
                    parameter_y_array[i] = parameters[parameter_y_index][1] + i * parameters[parameter_y_index][2]

                # set the values for this index
                for i in range(parameters_len):
                    info_dic[parameters[i][0]] = parameters[i][1] + base_indexes[i] * parameters[i][2]

                amplitude = info_dic['amplitude']
                delay = info_dic['delay']
                width = info_dic['width']
                delta = info_dic['delta']

                # create the labels
                label_dict = {
                    'amplitude': 'amplitude = ' + str(round(amplitude, 3)) + 'meV ',
                    'delay': 'delay = ' + str(round(delay, 3)) + 'ps ',
                    'width': 'width = ' + str(round(width, 3)) + 'ps ',
                    'delta': 'delta = ' + str(round(delta, 3)) + 'meV '
                }

                # remove the parameters labels
                del label_dict[parameter_x]
                del label_dict[parameter_y]

                # creat the whole label
                values = label_dict.values()
                label = ', '.join(values)

                # create density plot
                fig = plt.figure(figsize=[10.0, 10.0])
                title_font = {'size': 30}
                font = {'size': 15}

                mpl.rc('font', **title_font)
                fig.suptitle('Density plot (' + parameter_x + ' vs ' + parameter_y + ')')
                mpl.rc('font', **font)
                if surface:
                    ax = fig.add_subplot(111, projection='3d')
                else:
                    ax = fig.add_subplot(111)
                # Create the plot and colour palette
                if surface:
                    X, Y = np.meshgrid(parameter_x_array, parameter_y_array)
                    ax.plot_surface(X, Y, plot_states,
                                    vmin=0.0, vmax=1.0, cmap='viridis')
                    fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax)
                elif infidelity:
                    max_value = np.amax(plot_states)
                    ax.pcolormesh(parameter_x_array, parameter_y_array, plot_states,
                                  vmin=0.0, vmax=max_value, cmap='viridis', shading='auto')
                    norm = mpl.colors.Normalize(vmin=truncate_value, vmax=max_value)
                    fig.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=norm), ax=ax)
                elif truncate:
                    ax.pcolormesh(parameter_x_array, parameter_y_array, plot_states,
                                  vmin=0.0, vmax=truncate_value, cmap='tab20', shading='auto')
                    norm = mpl.colors.Normalize(vmin=truncate_value, vmax=1.0)
                    fig.colorbar(plt.cm.ScalarMappable(cmap='tab20', norm=norm), ax=ax)
                else:
                    ax.pcolormesh(parameter_x_array, parameter_y_array, plot_states,
                                  vmin=0.0, vmax=1.0, cmap='viridis', shading='auto')
                    fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax)

                ax.set_title(label)
                ax.set_xlabel(parameter_x)
                ax.set_ylabel(parameter_y)
                plt.show()

            else:
                print('please enter valid parameters')
        else:
            print('please enter parameter names as a strings')

    def trace_distance(self,
                       initial_state1=np.array([
                           [1.0 + 0.j, 0. + 0.j, 0. + 0.j],
                           [0. + 0.j, 0. + 0.j, 0. + 0.j],
                           [0. + 0.j, 0. + 0.j, 0. + 0.j]]),
                       initial_state2=np.array([
                           [0. + 0.j, 0. + 0.j, 0. + 0.j],
                           [0. + 0.j, 0. + 0.j, 0. + 0.j],
                           [0. + 0.j, 0. + 0.j, 1. + 0.j]]),
                       amplitude=1.0,
                       delay=1.5,
                       width=1.5,
                       delta=0.0
                       ):
        """
        Create plots between two initial states with the same parameters

        Parameters
        ---
        initial_state1 : np.array
            First initial state
        initial_state2 : np.array
            Second initial state
        amplitude : float
            Amplitude of beam in meV
        delay : float
            Delay between the two pulses, counter intuitive ordering
        width : float
            Width of the beams
        delta : float
            Detuning of the pulses
        """
        # compute dynamics for the two initial states
        # load data from the process tensor
        info_dict = self.process_tensor.description_dict['pt_tempo_description_dict']
        time_max = info_dict['time_max']
        dt = info_dict['dt']

        # setup gaussian pulse
        def gaussian(t_i, amplitude_i, delay_i, width_i):
            return amplitude_i * np.exp(-((t_i - (time_max / 2) - delay_i) ** 2) / (2 * (width_i ** 2)))

        # setup pulses and hamiltonian
        def omega_p(t_i):
            return gaussian(t_i, amplitude * 1.519, delay / 2, width)

        def omega_s(t_i):
            return gaussian(t_i, amplitude * 1.519, -delay / 2, width)

        def hamiltonian(t_i):
            return 0.5 * np.array(
                [[0. + 0.j, omega_p(t_i) + 0.j, 0. + 0.j],
                 [omega_p(t_i) + 0.j, 2 * delta * 1.519 + 0.j, omega_s(t_i) + 0.j],
                 [0. + 0.j, omega_s(t_i) + 0.j, 0. + 0.j]])

        # setup the system
        system = tempo.TimeDependentSystem(hamiltonian)

        # iterate over the initial states, calculating the value at each point
        initial_states = [initial_state1, initial_state2]
        states = []
        for initial_state in initial_states:
            dynamics = self.process_tensor.compute_dynamics_from_system(system=system, initial_state=initial_state)
            states.append(dynamics.states)

        # define trace distance
        def trace_norm(rho) -> float:
            """
            Trace norm of a density matrix. This is also the Schatten norm for p=1.
            """
            return np.real(np.trace(scipy.linalg.sqrtm(rho.transpose().conjugate() @ rho)))

        def trace_distance(rho, sigma) -> float:
            """
            The distance induced by the trace norm.
            """
            return 0.5 * trace_norm(rho - sigma)

        # calculate the trace distance
        trace_dis_array = []
        for t in range(int(time_max / dt) + 1):
            trace_dis_array.append(trace_distance(states[0][t], states[1][t]))

        # calculate the differential trace distance
        dif_trace_dis_array = []
        for t in range(int(time_max / dt)):
            dif_trace_dis_array.append((trace_dis_array[t + 1] - trace_dis_array[t]) / dt)

        # create time array
        time_array = np.arange(0.0, time_max + dt / 2, dt)

        # create the plot
        fig = plt.figure(figsize=[10.0, 10.0])
        title_font = {'size': 30}
        font = {'size': 15}

        mpl.rc('font', **title_font)
        fig.suptitle('Trace distance')
        mpl.rc('font', **font)

        ax = fig.add_subplot(211)
        ax.plot(time_array, trace_dis_array)
        plt.grid(True, axis='x')
        ax.set_ylabel('Trace Distance')
        ax.set_title('delay = ' + str(round(delay, 3)) + '(ps), width = ' +
                     str(round(width, 3)) + '(ps), amplitude = ' +
                     str(round(amplitude, 3)) + '(meV)  delta = ' +
                     str(round(delta, 3)) + '(meV)')

        ax.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off

        ax2 = fig.add_subplot(212)
        ax2.plot(time_array[0:-1], dif_trace_dis_array)
        plt.grid(True, axis='x')
        ax2.set_xlabel('Time (ps)')
        ax2.set_ylabel('Differential trace distance')

        plt.show()

    def change_states_basis(self,
                            U):
        """
                Change the basis of the states

                Parameters
                ---
                U : the unitary transform matrix
                """

        states = self.states[0]
        info = self.states[1]
        parameters = info['parameters']
        steps = []
        for parameter in parameters:
            steps.append(parameter[3])

        # get number of parameters and setup the index array
        num_parameters = len(parameters)
        index_array = np.repeat(0, num_parameters)

        new_states = np.zeros(tuple(steps), dtype=list)

        done = False
        while not done:
            state = states[tuple(index_array)]
            new_states[tuple(index_array)] = []
            for state_t in state:
                new_state_t = np.transpose(np.conjugate(U)) @ state_t @ U
                new_states[tuple(index_array)].append(new_state_t)

            # increment the index
            index_array[0] += 1
            for i in range(num_parameters):
                if index_array[i] == steps[i] and i != num_parameters - 1:
                    index_array[i + 1] += 1
                    index_array[i] = 0
                elif index_array[i] == steps[i]:
                    done = True

        self.states = [new_states, info]
