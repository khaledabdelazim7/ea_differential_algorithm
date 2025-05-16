import os
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

#Data Loading & Preprocessing 
def load_data():
    (x_tr, y_tr), (x_te, y_te) = mnist.load_data()
    x_tr = x_tr.reshape(-1,784).astype('float32') / 255.0
    x_te = x_te.reshape(-1,784).astype('float32') / 255.0
    y_tr_cat = to_categorical(y_tr, 10).astype('float32')
    y_te_cat = to_categorical(y_te, 10).astype('float32')
    return x_tr, y_tr_cat, x_te, y_te, y_te_cat

#Model Definition
def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax', name='head')
    ])

#Pretrain & Freeze
def pretrain_and_freeze(x, y, epochs=5, bs=64):
    y = y.astype('float32')
    m = build_model()
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    m.fit(x, y, epochs=epochs, batch_size=bs, verbose=0, validation_split=0.1)
    for layer in m.layers:
        if layer.name != 'head':
            layer.trainable = False
    return m

#Head Weight Utilities
def get_head_weights(model):
    w, b = model.get_layer('head').get_weights()
    return np.concatenate([w.flatten(), b.flatten()])

def set_head_weights(model, vec):
    w, b = model.get_layer('head').get_weights()
    size_w = w.size
    new_w = vec[:size_w].reshape(w.shape)
    new_b = vec[size_w:].reshape(b.shape)
    model.get_layer('head').set_weights([new_w, new_b])

#Fitness (Head Only)
def evaluate_fitness(vec, x_sub, y_sub):
    set_head_weights(pretrained_model, vec)
    preds = pretrained_model(x_sub, training=False)
    loss = tf.keras.losses.categorical_crossentropy(y_sub, preds)
    acc = tf.keras.metrics.categorical_accuracy(y_sub, preds)
    return -tf.reduce_mean(loss).numpy(), tf.reduce_mean(acc).numpy()

def compute_diversity(pop):
    pairs = [(i,j) for i in range(len(pop)) for j in range(i+1,len(pop))]
    return np.mean([np.linalg.norm(pop[i]-pop[j]) for i,j in pairs]) if pairs else 0

def fitness_sharing(fs, pop, sigma=0.1):
    out = []
    for i,fi in enumerate(fs):
        niche = sum(1 - np.linalg.norm(pop[i]-other)/sigma
                    for other in pop if np.linalg.norm(pop[i]-other)<sigma)
        out.append(fi/(niche if niche>0 else 1))
    return out

#Xavier Init
def init_xavier(dim):
    # head vector dim = 16*10 + 10 = 170
    # but we can ignore shape: just uniform from [-limit,limit]
    limit = np.sqrt(6 / (16 + 10))
    return np.random.uniform(-limit, limit, size=dim)

#Differential Evolution on Head
def differential_evolution(x_sub, y_sub,
                           pop_size, generations, F, CR,
                           init_strategy, mutation_strategy,
                           crossover_strategy, selection_strategy,
                           diversity_method=None, hybrid=False):
    dim = head_vector.size

    def init_ind():
        if init_strategy == 'uniform':
            return np.random.uniform(-0.5, 0.5, dim)
        if init_strategy == 'xavier':
            return init_xavier(dim)
        # gaussian around pretrained head
        return head_vector + np.random.randn(dim) * 0.1

    pop = [init_ind() for _ in range(pop_size)]
    fitness = [evaluate_fitness(ind, x_sub, y_sub)[0] for ind in pop]
    logs = {'acc':[], 'loss':[], 'fit':[], 'div':[]}

    for gen in range(1, generations+1):
        new_pop = []
        for i in range(pop_size):
            idxs = list(range(pop_size)); idxs.remove(i)
            # mutation
            if mutation_strategy == 'rand1':
                a,b,c = random.sample(idxs,3)
                mutant = pop[a] + F * (pop[b] - pop[c])
            elif mutation_strategy == 'rand2':
                a,b,c,d,e = random.sample(idxs,5)
                mutant = pop[a] + F * ((pop[b] - pop[c]) + (pop[d] - pop[e]))
            elif mutation_strategy == 'best1':
                best = pop[np.argmax(fitness)]
                a,b   = random.sample(idxs,2)
                mutant = best + F * (pop[a] - pop[b])
            elif mutation_strategy == 'jde':
                if random.random() < 0.1: F = random.uniform(0.1,1)
                if random.random() < 0.1: CR = random.random()
                a,b,c = random.sample(idxs,3)
                mutant = pop[a] + F * (pop[b] - pop[c])
            else:
                mutant = pop[i] + np.random.normal(0, 0.1, dim)

            # crossover
            if crossover_strategy == 'binomial':
                mask = np.random.rand(dim) < CR
                if not mask.any(): mask[np.random.randint(dim)] = True
            else:  # exponential
                mask = np.zeros(dim, bool)
                
                start, L = np.random.randint(dim), 0
                while L<dim and np.random.rand()<CR:
                    mask[(start+L)%dim] = True; L+=1

            trial = np.where(mask, mutant, pop[i])
            if hybrid:
                trial += np.random.normal(0, 0.01, dim)

            # selection
            f_t, acc_t = evaluate_fitness(trial, x_sub, y_sub)
            if selection_strategy == 'greedy':
                if f_t > fitness[i]:
                    new_pop.append(trial); fitness[i] = f_t
                else:
                    new_pop.append(pop[i])
            else:  # tournament
                challenger = random.choice([trial, pop[i]])
                f_c, _ = evaluate_fitness(challenger, x_sub, y_sub)
                if f_c > fitness[i]:
                    new_pop.append(challenger); fitness[i] = f_c
                else:
                    new_pop.append(pop[i])

        pop = new_pop
        results = [evaluate_fitness(ind, x_sub, y_sub) for ind in pop]
        fits    = [r[0] for r in results]
        accs    = [r[1] for r in results]
        losses  = [-r[0] for r in results]
        if diversity_method == 'sharing':
            fits = fitness_sharing(fits, pop)

        logs['acc'].append(max(accs))
        logs['loss'].append(min(losses))
        logs['fit'].append(max(fits))
        logs['div'].append(compute_diversity(pop))

        print(f"Gen {gen}: Acc={logs['acc'][-1]:.4f} "
              f"Loss={logs['loss'][-1]:.4f} Fit={logs['fit'][-1]:.4f} "
              f"Div={logs['div'][-1]:.4f}")

    best_idx = int(np.argmax(fitness))
    best_vec = pop[best_idx]
    return logs, best_vec


#Backprop Training
def train_backprop(x_sub, y_sub, x_val, y_val, epochs=10, bs=128):
    model = build_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_sub, y_sub, validation_data=(x_val, y_val),
                        epochs=epochs, batch_size=bs, verbose=0)
    return model, history.history

#Display Results
def display_results(nb, logs_de, best_vec, x_te, y_te, logs_bp, preds_bp):
    for t in nb.tabs():
        nb.forget(t)

    # Tab1: Metrics Comparison
    fr1 = ttk.Frame(nb)
    nb.add(fr1, text="Metrics")
    fig1 = plt.Figure(figsize=(8,6))
    axs = fig1.subplots(2,2)
    axs[0,0].plot(logs_de['acc'], label='DE Acc')
    axs[0,0].plot(logs_bp['accuracy'], label='BP Acc')
    axs[0,0].set_title("Accuracy"); axs[0,0].legend(); axs[0,0].grid()
    axs[0,1].plot(logs_de['loss'], label='DE Loss')
    axs[0,1].plot(logs_bp['loss'], label='BP Loss')
    axs[0,1].set_title("Loss"); axs[0,1].legend(); axs[0,1].grid()
    axs[1,0].plot(logs_de['fit'], label='DE Fit')
    axs[1,0].plot(logs_bp['val_accuracy'], label='BP Val Acc')
    axs[1,0].set_title("Fitness / Val Acc"); axs[1,0].legend(); axs[1,0].grid()
    axs[1,1].plot(logs_de['div'], label='DE Div')
    axs[1,1].set_title("Diversity (DE)"); axs[1,1].legend(); axs[1,1].grid()
    fig1.tight_layout()
    FigureCanvasTkAgg(fig1, fr1).get_tk_widget().pack(fill='both', expand=True)

    # Tab2: Confusion Matrices
    fr2 = ttk.Frame(nb)
    nb.add(fr2, text="Confusion")
    fig2 = plt.Figure(figsize=(10,4))
    ax_de = fig2.add_subplot(1,2,1)
    ax_bp = fig2.add_subplot(1,2,2)
    set_head_weights(pretrained_model, best_vec)
    preds_de = pretrained_model(x_te, training=False).numpy().argmax(axis=1)
    cm_de = confusion_matrix(y_te, preds_de)
    ConfusionMatrixDisplay(cm_de).plot(ax=ax_de, cmap=plt.cm.Blues)
    ax_de.set_title("DE Confusion")
    cm_bp = confusion_matrix(y_te, preds_bp)
    ConfusionMatrixDisplay(cm_bp).plot(ax=ax_bp, cmap=plt.cm.Oranges)
    ax_bp.set_title("BP Confusion")
    fig2.tight_layout()
    FigureCanvasTkAgg(fig2, fr2).get_tk_widget().pack(fill='both', expand=True)

    # Tab3: Sample Predictions
    fr3 = ttk.Frame(nb)
    nb.add(fr3, text="Samples")
    fig3 = plt.Figure(figsize=(8,4))
    axs3 = fig3.subplots(2,10)
    for d in range(10):
        idx_de = np.where(y_te==d)[0][0]
        axs3[0,d].imshow(x_te[idx_de].reshape(28,28), cmap='gray')
        axs3[0,d].axis('off'); axs3[0,d].set_title(str(preds_de[idx_de]))
        idx_bp = np.where(y_te==d)[0][0]
        axs3[1,d].imshow(x_te[idx_bp].reshape(28,28), cmap='gray')
        axs3[1,d].axis('off'); axs3[1,d].set_title(str(preds_bp[idx_bp]))
    fig3.tight_layout()
    FigureCanvasTkAgg(fig3, fr3).get_tk_widget().pack(fill='both', expand=True)

#Main GUI
def run_gui():
    global pretrained_model, head_vector
    x_tr, y_tr_cat, x_te, y_te, y_te_cat = load_data()
    x_sub, y_sub = x_tr[:3000], y_tr_cat[:3000]
    x_val, y_val = x_tr[3000:3500], y_tr_cat[3000:3500]

    pretrained_model = pretrain_and_freeze(x_tr, y_tr_cat, epochs=5, bs=128)
    head_vector      = get_head_weights(pretrained_model)

    root = tk.Tk()
    root.title("DE vs BP MNIST Comparison")
    root.geometry("1200x700")
    style = ttk.Style(root); style.theme_use('clam')

    paned = ttk.PanedWindow(root, orient='horizontal')
    paned.pack(fill='both', expand=True)

    # Left controls
    ctrl = ttk.Frame(paned, padding=10, width=300)
    paned.add(ctrl, weight=0)
    strategies = [
        ("Init", ['uniform','xavier','normal']),
        ("Mutate", ['rand1','rand2','best1','jde','gauss']),
        ("Cross", ['binomial','exponential']),
        ("Select", ['greedy','tournament']),
        ("Diversity", ['none','sharing','crowding'])
    ]
    vars_ = {}
    for i,(lbl,opts) in enumerate(strategies):
        ttk.Label(ctrl, text=lbl).grid(row=i, column=0, sticky='w', pady=5)
        v = tk.StringVar(value=opts[0])
        vars_[lbl] = v
        ttk.Combobox(ctrl, textvariable=v, values=opts, state='readonly').grid(
            row=i, column=1, padx=5, pady=5, sticky='ew'
        )

    entries = [("Population","60"), ("Generations","100"), ("F","0.8"), ("CR","0.9")]
    ents = {}
    for j,(lbl,defv) in enumerate(entries, start=len(strategies)):
        ttk.Label(ctrl, text=lbl).grid(row=j, column=0, sticky='w', pady=5)
        e = ttk.Entry(ctrl)
        e.insert(0, defv)
        e.grid(row=j, column=1, padx=5, pady=5, sticky='ew')
        ents[lbl] = e

    def on_run():
        logs_de, best = differential_evolution(
            x_sub, y_sub,
            pop_size=int(ents['Population'].get()),
            generations=int(ents['Generations'].get()),
            F=float(ents['F'].get()),
            CR=float(ents['CR'].get()),
            init_strategy=vars_['Init'].get(),
            mutation_strategy=vars_['Mutate'].get(),
            crossover_strategy=vars_['Cross'].get(),
            selection_strategy=vars_['Select'].get(),
            diversity_method=vars_['Diversity'].get(),
            hybrid=True
        )
        bp_model, logs_bp = train_backprop(
            x_sub, y_sub, x_val, y_val,
            epochs=int(ents['Generations'].get()), bs=128
        )
        preds_bp = bp_model.predict(x_te).argmax(axis=1)
        display_results(notebook, logs_de, best, x_te, y_te, logs_bp, preds_bp)

    ttk.Button(ctrl, text="▶ Run", command=on_run, style='Accent.TButton').grid(
        row=j+1, column=0, columnspan=2, pady=15, sticky='ew'
    )
    ctrl.columnconfigure(1, weight=1)

    res_frame = ttk.Frame(paned, padding=5)
    paned.add(res_frame, weight=1)
    notebook = ttk.Notebook(res_frame)
    notebook.pack(fill='both', expand=True)

    root.mainloop() 

if __name__ == "__main__":
    run_gui()
