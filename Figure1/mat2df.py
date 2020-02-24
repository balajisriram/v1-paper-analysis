import scipy.io as sio
import pprint
import pandas as pd
import numpy as np
from collections.abc import Iterable

def mat2df(input_file):
    mat_data = sio.loadmat(input_file,struct_as_record=False,squeeze_me=True)

    ctr = mat_data['compiledTrialRecords']
    cd = mat_data['compiledDetails']
    lut = mat_data['compiledLUT']

    compiled_record = pd.DataFrame({'trial_number':ctr.trialNumber,
                                    'correction_trial':ctr.correctionTrial,
                                     'result':ctr.result,
                                     'proposed_penalty_duration':ctr.proposedPenaltyDuration,
                                     'did_stochastic_response':ctr.didStochasticResponse,
                                     'num_ports':ctr.numPorts,
                                     'response':ctr.response,
                                     'num_steps_in_protocol':ctr.numStepsInProtocol,
                                     'did_human_response':ctr.didHumanResponse,
                                     'contained_a_pause':ctr.containedAPause,
                                     'protocol_name':lut[ctr.protocolName-1],
                                     'stim_manager_class':lut[ctr.stimManagerClass-1],
                                     'distractor_ports':ctr.distractorPorts,
                                     'sound_on':ctr.soundOn,
                                     'training_step_name':lut[ctr.trainingStepName-1],
                                     'first_IRI':ctr.firstIRI,
                                     'reinforcement_manager_class':lut[ctr.reinforcementManagerClass-1],
#                                      'potential_reward_scalar':ctr.potentialRewardScalar,
                                     'session_number':ctr.sessionNumber,
                                     'correct':ctr.correct,
                                     'date':ctr.date,
                                     'correct':ctr.correct,
                                     'auto_version':ctr.autoVersion,
                                     'proposed_reward_duration':ctr.proposedRewardDuration,
                                     'scheduler_class':ctr.schedulerClass,
                                     'response_time':ctr.responseTime,
                                     'contained_forced_rewards':ctr.containedForcedRewards,
                                     'criterion_class':lut[ctr.criterionClass-1],
                                     'target_ports':ctr.targetPorts,
                                     'actual_reward_duration':ctr.actualRewardDuration,
                                     'num_requests':ctr.numRequests,
                                     'protocol_date':ctr.protocolDate,
#                                      'potential_penalty':ctr.potentialPenalty,
                                     'step':ctr.step,
#                                      'potential_request_reward_MS':ctr.potentialRequestRewardMS,
                                   })
    compiled_record.set_index('trial_number')
    
    if not isinstance(cd,Iterable):
        cd = [cd]

    for deets in cd:
        if deets.className=='images':
            deets_df = pd.DataFrame({'trial_number':deets.trialNums,
                                     'left_im':lut[deets.records.leftIm-1],
                                     'right_im':lut[deets.records.rightIm-1],
                                     'details_correction':deets.records.correctionTrial,
                                    })
            deets_df.set_index('trial_number')
            compiled_record = compiled_record.merge(deets_df,how='outer',on='trial_number')
        elif deets.className=='afcGratings':
            deets_df = pd.DataFrame({'trial_number':deets.trialNums,
                                     'drift_frequencies':deets.records.driftfrequencies,
                                     'max_duration':deets.records.maxDuration,
                                     'LED0':deets.records.LED[0],
                                     'LED1':deets.records.LED[1],
                                     'contrasts':deets.records.contrasts,
                                     'pct_correction_trials':deets.records.pctCorrectionTrials,
                                     'radii':deets.records.radii,
                                     'pix_per_cycs':deets.records.pixPerCycs,
                                     'orientations':deets.records.orientations,
                                     'afc_grating_type':deets.records.afcGratingType,
                                     'phases':deets.records.phases,
                                     'do_combos':deets.records.doCombos,
                                     'details_correction':deets.records.correctionTrial,
                                    })
            deets_df.set_index('trial_number')
            compiled_record = compiled_record.merge(deets_df,how='outer',on='trial_number')
        elif deets.className=='afcGratingsPDPhases':
            deets_df = pd.DataFrame({'trial_number':deets.trialNums,
                                     'drift_frequencies':deets.records.driftfrequencies,
                                     'max_duration':deets.records.maxDuration,
                                     'LED0':deets.records.LED[0],
                                     'LED1':deets.records.LED[1],
                                     'contrasts':deets.records.contrasts,
                                     'pct_correction_trials':deets.records.pctCorrectionTrials,
                                     'radii':deets.records.radii,
                                     'pix_per_cycs':deets.records.pixPerCycs,
                                     'orientations':deets.records.orientations,
                                     'afc_grating_type':deets.records.afcGratingType,
                                     'phases':deets.records.phases,
                                     'do_combos':deets.records.doCombos,
                                     'details_correction':deets.records.correctionTrial,
                                    })
            deets_df.set_index('trial_number')
            compiled_record = compiled_record.merge(deets_df,how='outer',on='trial_number')
        else:
            print('not processed for ', deets.className)


    return compiled_record