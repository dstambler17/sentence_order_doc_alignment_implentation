
import time
import torch
from doc_align_classifier.train.train_utils import *
from sklearn.metrics import f1_score

##Define Loss and Accuracy Eval Function
def eval_acc_and_loss_func(model, loader, device, loss_metric, is_train = False, verbose = 1):
    '''
    Evaluate Function for CNN training
    Slightly different than eval function from part 1
    '''
    correct, total, loss_sum = 0, 0, 0
    
    #temp_idx = 0  TODO: remove when down with loop debug
    
    eval_type = "Train" if is_train else "Validation"
    #Declare f1 score info
    f1_s = 0
    idx = 0
    
    for X_1, X_2, Y in loader:
        outputs, predicted, calculated_loss = None, None, None
        X_1, X_2, Y = X_1.to(device), X_2.to(device), Y.to(device)
        
        X_1, X_2 = X_1.float(), X_2.float()
        outputs = model(X_1, X_2)
        
        #Reshape output and turn y into a float
        outputs = outputs.view(-1)
        Y = Y.float()
        predicted = torch.round(outputs)
        total += Y.size(0)
        
        correct += (predicted == Y).sum().item()
        calculated_loss = loss_metric(outputs,Y).item()
        loss_sum += calculated_loss
        
        #Update f1_score
        f1_s += f1_score(Y.detach(), predicted.detach(), labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
        idx += 1
    
    f1_s = (f1_s / idx)
    outputs, predicted, calculated_loss = None, None, None
    if verbose:
        print('%s accuracy: %f %%' % (eval_type, 100.0 * correct / total))
        print('%s f1_score: %f %%' %  (eval_type, f1_s) )
        print('%s loss: %f' % (eval_type, loss_sum / total))
    print
    return (100.0 * correct / total), f1_s, loss_sum/ total

#DEFINE TRAIN LOOP HERE
def train(model,
          optimizer,
          loss_metric,
          lr,
          train_dataloader,
          valid_dataloader,
          device,
          epochs=50,
          stopping_threshold=5,
          saving_per_epoch=1,
          base_save_path="/home/dstambl2/doc_alignment_implementations/thompson_2021_doc_align/devtools/models_fast_train",
          model_name="train_loop_model_fast",
          load_train_hist_path=None,
          **kwargs):
    """
    For each epoch, loop through batch,
    compute forward and backward passes, apply gradient updates
    Evaluate results and output
    """

    #If data already exists, that means the model was preloaded and training should
    #resume from where it was interupted
    if load_train_hist_path is not None:
      val_f1_store, val_loss_store, train_loss_store, train_f1_store, start_epoch =load_train_data(load_train_hist_path)
    else:
      train_loss_store, train_f1_store = [], []
      val_loss_store, val_f1_store, = [], []
      start_epoch = 0

    #Declare variables for early stopping
    last_val_f1, stop_tracker = 0, 0

    #training loop:
    print("Starting Training")
    for epoch in range(start_epoch, epochs):
      time1 = time.time() #timekeeping
      outputs, loss = None, None

      model.train()
      
      idx, f1_s, correct_train, total, loss_sum = 0, 0, 0, 0, 0     
      for i, (x_1, x_2, y) in enumerate(train_dataloader):
        
        # Print device human readable names
        #torch.cuda.get_device_name()

        x_1, x_2, y = x_1.to(device), x_2.to(device), y.to(device)

        #loss calculation and gradient update:

        x_1, x_2 = x_1.float(), x_2.float()
        if i > 0 or epoch > 0:
          optimizer.zero_grad()
        outputs = model.forward(x_1, x_2)
        
        #Reshape output and turn y into a float
        outputs = outputs.view(-1)
        y = y.float()
        #print(y.shape, outputs.shape, outputs, "Loss inp info")

        
        loss = loss_metric(outputs, y)
        loss.backward()
                      
        ##performing update:
        optimizer.step()

        #Update Loss Info
        loss_sum += loss.item()
        
        #Acc was likely not increasing, because preds kept rounding to zero
        predicted = torch.round(outputs) #NOTE: MODEL predictions keep rounding to zero
        #print(outputs, predicted, y, "predicted stuff", y.size(0))

        total += y.size(0)
        correct_train += (predicted == y).sum().item()
        
        #Update f1_score
        f1_s += f1_score(y.detach(), predicted.detach(), labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
        idx = i
              
      print("Epoch",epoch+1,':')

      f1_train = (f1_s/idx)
      model.eval()
      with torch.no_grad():
        #Print Train Info
        print('%s accuracy: %f %%' % ("Train", 100.0 * (correct_train / total)))
        print('%s f1_score: %f %%' %  ("Train", f1_train) )
        print('%s loss: %f' % ("Train", loss_sum / total))
        print
        
        train_acc, train_loss = 100.0 * correct_train / total, loss_sum/ total
        val_acc, val_f1, val_loss = eval_acc_and_loss_func(model, valid_dataloader, device, loss_metric, is_train = False)

        val_f1_store.append(val_f1)
        val_loss_store.append(val_loss)

        train_loss_store.append(train_loss)
        train_f1_store.append(f1_train)

      time2 = time.time() #timekeeping
      #if show_progress:
      print('Elapsed time for epoch:',time2 - time1,'s')
      print('ETA of completion:',(time2 - time1)*(epochs - epoch - 1)/60,'minutes')
      
      if (epoch + 1) % saving_per_epoch == 0:
        handle_model_save(val_f1_store, val_loss_store, train_loss_store, train_f1_store,
                          base_save_path, model_name, epoch + 1, model, optimizer)
        print("Model Copy Saved")
      print()


      #Handle early stopping logic
      #if val_loss >= last_val_loss:
      if val_f1 <= last_val_f1:
            stop_tracker += 1
            if stop_tracker >= stopping_threshold:
                print('Early Stopping triggered, Convergence has occured')
                plot_loss_charts(train_loss_store, val_loss_store)
                plot_f1_score(train_f1_store, val_f1_store)
                handle_model_save(val_f1_store, val_loss_store, train_loss_store, train_f1_store,
                  base_save_path, model_name, epoch + 1, model, optimizer)
                print("Model Copy Saved")

                return train_loss_store, val_f1_store
      else:
          stop_tracker = 0
      last_val_f1 = val_f1


    plot_loss_charts(train_loss_store, val_loss_store)
    plot_f1_score(train_f1_store, val_f1_store)
    return train_loss_store, val_f1_store
