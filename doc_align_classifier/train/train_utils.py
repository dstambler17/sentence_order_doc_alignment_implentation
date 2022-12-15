import json
import torch
import matplotlib.pyplot as plt

#Plotting/training utils
def plot_loss_charts(train_loss_store, validation_loss_store):
  '''
  Plots loss charts over course of training
  '''
  ## Plotting epoch-wise test loss curve:
  plt.plot(train_loss_store, '-o', label = 'train_loss', color = 'orange')
  plt.plot(validation_loss_store, label = 'validation_loss', color = 'blue')
  plt.xlabel('Epoch Number')
  plt.ylabel('Loss At each epoch')
  plt.legend()
  plot_loss_charts
  #plt.show()
  plt.savefig('/home/dstambl2/doc_alignment_implementations/thompson_2021_doc_align/devtools/plots/%s_fast_train.png' % "loss_plots")
  plt.clf()


def plot_f1_score(train_score_store, validation_score_store, skip_plot=False):
  '''
  Plots Accuracy charts over course of training
  '''
  #Don't plot if this flag is set to true
  if skip_plot:
    return

  ## Plotting epoch-wise test loss curve:
  plt.plot(train_score_store, '-o', label = 'train_f1_score', color = 'orange')
  plt.plot(validation_score_store, label = 'validation_f1_score', color = 'blue')
  plt.xlabel('Epoch Number')
  plt.ylabel('F1 Score At each epoch')
  plt.legend()
  #plt.show()
  plt.savefig('/home/dstambl2/doc_alignment_implementations/thompson_2021_doc_align/devtools/plots/%s_fast_train.png' % "f1_score_plots")
  plt.clf()



# %%
##Define Model Saving functionality
def handle_model_save(val_acc_store, val_loss_store, train_loss_store, train_acc_store,
                      base_path, model_name, epoch_num, model, optimizer):
  '''
  Function for saving models and training data
  '''
  save_dict = {
      'val_acc_store': val_acc_store,
      'val_loss_store': val_loss_store,
      'train_loss_store': train_loss_store,
      'train_acc_store': train_acc_store,
      'epoch': epoch_num,
  }
  traing_info_path = "%s/%s_epoch_%s.json" % (base_path, model_name, epoch_num)
  with open(traing_info_path, 'w') as f:
    json.dump(save_dict, f)

  MODEL_PATH = "%s/%s_%s" % (base_path, model_name, epoch_num)
  torch.save({
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              }, MODEL_PATH)


#Define loading historic training data func
def load_train_data(traing_info_path):
  with open(traing_info_path) as json_file:
    data = json.load(json_file)
  
  return data['val_acc_store'], data['val_loss_store'], data['train_loss_store'], data['train_acc_store'], int(data['epoch'])
