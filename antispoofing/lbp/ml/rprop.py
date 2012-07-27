#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.dos.anjos@gmail.com>
# Tue 16 Aug 14:39:22 2011 

"""Trains an MLP using RProp
"""

import sys
import bob
import numpy
import numpy.linalg as la
from . import perf

class Analyzer:
  """Can analyze results in the end of a run. It can also save itself"""

  def gentargets(self, data, target):
    t = numpy.vstack(data.shape[0] * (target,))
    return t, numpy.empty_like(t)

  def __init__(self, train, devel, target):

    self.train = train
    self.devel = devel
    self.target = target

    real_train = self.gentargets(train[0], target[0])
    attack_train = self.gentargets(train[1], target[1])
    real_devel = self.gentargets(devel[0], target[0])
    attack_devel = self.gentargets(devel[1], target[1])

    self.train_target = (real_train[0], attack_train[0])
    self.train_output = (real_train[1], attack_train[1])
    self.devel_target = (real_devel[0], attack_devel[0])
    self.devel_output = (real_devel[1], attack_devel[1])

    self.data = {} #where to store variables that will be saved
    self.data['epoch'] = []
    self.data['real-train-rmse'] = []
    self.data['attack-train-rmse'] = []
    self.data['real-devel-rmse'] = []
    self.data['attack-devel-rmse'] = []
    self.data['train-far'] = []
    self.data['train-frr'] = []
    self.data['devel-far'] = []
    self.data['devel-frr'] = []

  def __call__(self, machine, iteration):
    """Computes current outputs and evaluate performance"""

    def evalperf(outputs, targets):
      return la.norm(bob.measure.rmse(outputs, targets))

    for k in range(len(self.train)):
      machine(self.train[k], self.train_output[k])
      machine(self.devel[k], self.devel_output[k])

    self.data['real-train-rmse'].append(evalperf(self.train_output[0],
      self.train_target[0]))
    self.data['attack-train-rmse'].append(evalperf(self.train_output[1],
      self.train_target[1]))
    self.data['real-devel-rmse'].append(evalperf(self.devel_output[0],
      self.devel_target[0]))
    self.data['attack-devel-rmse'].append(evalperf(self.devel_output[1],
      self.devel_target[1]))

    thres = bob.measure.eer_threshold(self.train_output[1][:,0],
        self.train_output[0][:,0])
    train_far, train_frr = bob.measure.farfrr(self.train_output[1][:,0], 
        self.train_output[0][:,0], thres)
    devel_far, devel_frr = bob.measure.farfrr(self.devel_output[1][:,0],
        self.devel_output[0][:,0], thres)

    self.data['train-far'].append(train_far)
    self.data['train-frr'].append(train_frr)
    self.data['devel-far'].append(devel_far)
    self.data['devel-frr'].append(devel_frr)

    self.data['epoch'].append(iteration)

  def str_header(self):
    """Returns the string header of what I can print"""
    return "iteration: RMSE:real/RMSE:attack (EER:%) ( train | devel )"

  def __str__(self):
    """Returns a string representation of myself"""

    retval = "%d: %.4e/%.4e (%.2f%%) | %.4e/%.4e (%.2f%%)" % \
        (self.data['epoch'][-1], 
            self.data['real-train-rmse'][-1], 
            self.data['attack-train-rmse'][-1],
            50*(self.data['train-far'][-1] + self.data['train-frr'][-1]),
            self.data['real-devel-rmse'][-1],
            self.data['attack-devel-rmse'][-1],
            50*(self.data['devel-far'][-1] + self.data['devel-frr'][-1]),
            )
    return retval

  def save(self, f):
    """Saves my contents on the bob.io.HDF5File you give me."""

    for k, v in self.data.items(): f.set(k, numpy.array(v))

  def load(self, f):
    """Loads my contents from the bob.io.HDF5File you give me."""

    for k in f.paths():
      self.data[k.strip('/')] = f.read(k)

  def report(self, machine, test, pdffile, textfile):
    """Complete analysis of the contained data, with plots and all..."""

    import matplotlib; matplotlib.use('pdf') #avoids TkInter threaded start
    import matplotlib.pyplot as mpl
    from matplotlib.backends.backend_pdf import PdfPages

    real_test = self.gentargets(test[0], self.target[0])
    attack_test = self.gentargets(test[1], self.target[1])
    test_target = (real_test[0], attack_test[0])
    test_output = (real_test[1], attack_test[1])

    for k in range(len(self.train)):
      machine(self.train[k], self.train_output[k])
      machine(self.devel[k], self.devel_output[k])
      machine(test[k], test_output[k])

    # Here we start with the plotting and writing of tables in files
    # --------------------------------------------------------------

    perftable, eer_thres, mhter_thres = perf.performance_table(test_output, 
        self.devel_output, "Performance Table")

    devel_res, test_res = perf.perf_hter_thorough(test_output, self.devel_output, bob.measure.eer_threshold) # returns FAR/FRR/HTER for the development and test set

    tf = open(textfile, 'wt')
    tf.write(perftable)
    tf.close()

    pp = PdfPages(pdffile)

    fig = mpl.figure()
    perf.score_distribution_plot(test_output, self.devel_output,
        self.train_output, self.data['epoch'][-1], 50, eer_thres, mhter_thres)
    pp.savefig(fig)
    fig = mpl.figure()
    perf.roc(test_output, self.devel_output, self.train_output, 100,
        eer_thres, mhter_thres)
    pp.savefig(fig)
    fig = mpl.figure()
    perf.det(test_output, self.devel_output, self.train_output, 100,
        eer_thres, mhter_thres)
    pp.savefig(fig)
    fig = mpl.figure()
    perf.epc(test_output, self.devel_output, 100)
    pp.savefig(fig)
    fig = mpl.figure()
    perf.plot_rmse_evolution(self.data)
    pp.savefig(fig)
    fig = mpl.figure()
    perf.plot_eer_evolution(self.data)
    pp.savefig(fig)
    fig = mpl.figure()
    perf.evaluate_relevance(test, self.devel, self.train, machine)
    pp.savefig(fig)
    pp.close()
    return devel_res, test_res

def make_mlp(train, devel, batch_size, nhidden, epoch, max_iter=0,
    no_improvements=0, verbose=False):
  """Creates a randomly initialized MLP and train it using the input data.
  
  This method will create an MLP with a single hidden layer containing the
  number of hidden neurons indicated by the parameter `nhidden`. Then it will
  initialize the MLP with random weights and biases and train it for as long as
  the development shows improvement and will stop as soon as it does not
  anymore or we reach the maximum number of iterations.

  Performance is evaluated both on the trainining and development set during
  the training, every 'epoch' training steps. Each training step is composed of
  `batch_size` elements drawn randomly from all classes available in train set.

  Keyword Parameters:

  train
    An iterable (tuple or list) containing two arraysets: the first contains
    the real accesses (target = +1) and the second contains the attacks (target
    = -1).

  devel
    An iterable (tuple or list) containing two arraysets: the first contains
    the real accesses (target = +1) and the second contains the attacks (target
    = -1).

  batch_size
    An integer defining the number of samples per training iteration. Good
    values are greater than 100.

  nhidden
    Number of hidden neurons in the hidden layer.

  epoch
    The number of training steps to wait until we measure the error.

  max_iter
    If given (and different than zero), should tell us the maximum number of
    training steps to train the network for. If set to 0 just train until the
    development sets reaches the valley (in RMSE terms).

  no_improvements
    If given (and different than zero), should tell us the maximum number of
    iterations we should continue trying for in case we have no more
    improvements on the development set average RMSE term. This value, if set,
    should not be too small as this may cause a too-early stop. Values in the
    order of 10% of the max_iter should be fine.

  verbose
    Makes the training more verbose
  """

  VALLEY_CONDITION = 0.8 #of the minimum devel. set RMSE detected so far

  def stop_condition(min_devel_rmse, devel_rmse):
    """This method will detect a valley in the devel set RMSE"""

    return (VALLEY_CONDITION * devel_rmse) > min_devel_rmse


  target = [
      numpy.array([+1], 'float64'),
      numpy.array([-1], 'float64'),
      ]

  if verbose: print "Preparing analysis framework..."
  analyze = Analyzer(train, devel, target)

  if verbose: print "Setting up training infrastructure..."
  shuffler = bob.trainer.DataShuffler(train, target)
  shuffler.auto_stdnorm = True

  shape = (shuffler.data_width, nhidden, 1)
  machine = bob.machine.MLP(shape)
  machine.activation = bob.machine.Activation.TANH
  machine.randomize()
  machine.input_subtract, machine.input_divide = shuffler.stdnorm()

  trainer = bob.trainer.MLPRPropTrainer(machine, batch_size)
  trainer.trainBiases = True

  continue_training = True
  iteration = 0
  min_devel_rmse = sys.float_info.max
  best_machine = bob.machine.MLP(machine) #deep copy
  best_machine_iteration = 0

  # temporary training data selected by the shuffer
  shuffled_input = numpy.ndarray((batch_size, shuffler.data_width), 'float64')
  shuffled_target = numpy.ndarray((batch_size, shuffler.target_width), 'float64')

  if verbose: print analyze.str_header()

  try:
    while continue_training:

      analyze(machine, iteration)
      
      if verbose: print analyze

      avg_devel_rmse = (analyze.data['real-devel-rmse'][-1] + \
          analyze.data['attack-devel-rmse'][-1])/2

      if avg_devel_rmse < min_devel_rmse: #save best network, record minima
        best_machine_iteration = iteration
        best_machine = bob.machine.MLP(machine) #deep copy
        if verbose: print "%d: Saving best network so far with average devel. RMSE = %.4e" % (iteration, avg_devel_rmse)
        min_devel_rmse = avg_devel_rmse
        if verbose: print "%d: New valley stop threshold set to %.4e" % \
            (iteration, avg_devel_rmse/VALLEY_CONDITION)

      if stop_condition(min_devel_rmse, avg_devel_rmse):
        if verbose: 
          print "%d: Stopping on devel valley condition" % iteration
          print "%d: Best machine happened on iteration %d with average devel. RMSE of %.4e" % (iteration, best_machine_iteration, min_devel_rmse)

        break

      for i in range(epoch): #train for 'epoch' times w/o stopping for tests

        shuffler(shuffled_input, shuffled_target)
        trainer.train(machine, shuffled_input, shuffled_target)
        iteration += 1

      if max_iter > 0 and iteration > max_iter:
        if verbose: 
          print "%d: Stopping on max. iterations condition" % iteration
          print "%d: Best machine happened on iteration %d with average devel. RMSE of %.4e" % (iteration, best_machine_iteration, min_devel_rmse)
        break

      if no_improvements > 0 and \
          (iteration-best_machine_iteration) > no_improvements:
        if verbose:
          print "%d: Stopping because did not observe MLP performance improvements for %d iterations" % (iteration, iteration-best_machine_iteration)
          print "%d: Best machine happened on iteration %d with average devel. RMSE of %.4e" % (iteration, best_machine_iteration, min_devel_rmse)
        break

  except KeyboardInterrupt:
    if verbose:
      print "%d: User interruption captured - exiting in a clean way" % \
          iteration
      print "%d: Best machine happened on iteration %d with average devel. RMSE of %.4e" % (iteration, best_machine_iteration, min_devel_rmse)

    analyze(machine, iteration)

  return best_machine, analyze
