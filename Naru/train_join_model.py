import argparse
import collections
import glob
import os
import pprint
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

import common
import datasets
import estimators as estimators_lib
import experiments
import factorized_sampler
import join_utils
import made
import transformer
import utils
import train_utils
from update_utils.torch_util import get_torch_device

def create_parser():
    parser = argparse.ArgumentParser()

    # Training.
    parser.add_argument('--run',
                    nargs='+',
                    default=experiments.TEST_CONFIGS.keys(),
                    type=str,
                    required=False,
                    help='List of experiments to run.')
    parser.add_argument("--model", type=str, choices=["naru", "transformer"], default="naru", help="Training model")
    parser.add_argument("--training_type", type=str, choices=["train", "retrain"], default="train", help="Training type")
    parser.add_argument("--dataset", type=str, default="dmv-tiny", help="Dataset.")
    parser.add_argument("--num-gpus", type=int, default=0, help="#gpus.")
    parser.add_argument("--bs", type=int, default=1024, help="Batch size.")
    parser.add_argument(
        "--warmups",
        type=int,
        default=0,
        help="Learning rate warmup steps.  Crucial for Transformer.",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of epochs to train for."
    )
    parser.add_argument("--constant-lr", type=float, default=None, help="Constant LR?")
    parser.add_argument(
        "--column-masking",
        action="store_true",
        help="Column masking training, which permits wildcard skipping"
        " at querying time.",
    )

    # MADE.
    parser.add_argument(
        "--fc-hiddens", type=int, default=256, help="Hidden units in FC."
    )
    parser.add_argument("--layers", type=int, default=4, help="# layers in FC.")
    parser.add_argument("--residual", action="store_true", help="ResMade?")
    parser.add_argument("--direct-io", action="store_true", help="Do direct IO?")
    parser.add_argument(
        "--inv-order",
        action="store_true",
        help="Set this flag iff using MADE and specifying --order. Flag --order "
        "lists natural indices, e.g., [0 2 1] means variable 2 appears second."
        "MADE, however, is implemented to take in an argument the inverse "
        "semantics (element i indicates the position of variable i).  Transformer"
        " does not have this issue and thus should not have this flag on.",
    )
    parser.add_argument(
        "--input-encoding",
        type=str,
        default="binary",
        help="Input encoding for MADE/ResMADE, {binary, one_hot, embed}.",
    )
    parser.add_argument(
        "--output-encoding",
        type=str,
        default="one_hot",
        help="Iutput encoding for MADE/ResMADE, {one_hot, embed}.  If embed, "
        "then input encoding should be set to embed as well.",
    )

    # Transformer.
    parser.add_argument(
        "--heads",
        type=int,
        default=0,
        help="Transformer: num heads.  A non-zero value turns on Transformer"
        " (otherwise MADE/ResMADE).",
    )
    parser.add_argument(
        "--blocks", type=int, default=2, help="Transformer: num blocks."
    )
    parser.add_argument("--dmodel", type=int, default=32, help="Transformer: d_model.")
    parser.add_argument("--dff", type=int, default=128, help="Transformer: d_ff.")
    parser.add_argument(
        "--transformer-act", type=str, default="gelu", help="Transformer activation."
    )

    # Ordering.
    parser.add_argument(
        "--num-orderings", type=int, default=1, help="Number of orderings."
    )
    parser.add_argument(
        "--order",
        nargs="+",
        type=int,
        required=False,
        help="Use a specific ordering.  "
        "Format: e.g., [0 2 1] means variable 2 appears second.",
    )
    return parser

args = create_parser().parse_args()

if args.training_type == "retrain":
    DEVICE = get_torch_device(extra=True)
else:
    DEVICE = get_torch_device(extra=False)

def RunEpoch(
    split,
    model,
    opt,
    train_data,
    val_data=None,
    batch_size=100,
    upto=None,
    epoch_num=None,
    verbose=False,
    log_every=10,
    return_losses=False,
    table_bits=None,
):
    torch.set_grad_enabled(split == "train")
    model.train() if split == "train" else model.eval()
    dataset = train_data if split == "train" else val_data
    losses = []

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=(split == "train")
    )

    # How many orderings to run for the same batch?
    nsamples = 1
    if hasattr(model, "orderings"):
        nsamples = len(model.orderings)

    for step, xb in enumerate(loader):
        if split == "train":
            base_lr = 8e-4
            for param_group in opt.param_groups:
                if args.constant_lr:
                    lr = args.constant_lr
                elif args.warmups:
                    t = args.warmups
                    d_model = model.embed_size
                    global_steps = len(loader) * epoch_num + step + 1
                    lr = (d_model**-0.5) * min(
                        (global_steps**-0.5), global_steps * (t**-1.5)
                    )
                else:
                    lr = 1e-2

                param_group["lr"] = lr

        if upto and step >= upto:
            break

        xb = xb.to(DEVICE).to(torch.float32)

        # Forward pass, potentially through several orderings.
        xbhat = None
        model_logits = []
        num_orders_to_forward = 1
        if split == "test" and nsamples > 1:
            # At test, we want to test the 'true' nll under all orderings.
            num_orders_to_forward = nsamples

        for i in range(num_orders_to_forward):
            if hasattr(model, "update_masks"):
                # We want to update_masks even for first ever batch.
                model.update_masks()

            model_out = model(xb)
            model_logits.append(model_out)
            if xbhat is None:
                xbhat = torch.zeros_like(model_out)
            xbhat += model_out

        if model.input_bins is None:
            # NOTE: we have to view() it in this order due to the mask
            # construction within MADE.  The masks there on the output unit
            # determine which unit sees what input vars.
            xbhat = xbhat.view(-1, model.nout // model.nin, model.nin)
            # Equivalent to:
            loss = F.cross_entropy(xbhat, xb.long(), reduction="none").sum(-1).mean()
        else:
            if num_orders_to_forward == 1:
                loss = model.nll(xbhat, xb).mean()
            else:
                # Average across orderings & then across minibatch.
                #
                #   p(x) = 1/N sum_i p_i(x)
                #   log(p(x)) = log(1/N) + log(sum_i p_i(x))
                #             = log(1/N) + logsumexp ( log p_i(x) )
                #             = log(1/N) + logsumexp ( - nll_i (x) )
                #
                # Used only at test time.
                logps = []  # [batch size, num orders]
                assert len(model_logits) == num_orders_to_forward, len(model_logits)
                for logits in model_logits:
                    # Note the minus.
                    logps.append(-model.nll(logits, xb))
                logps = torch.stack(logps, dim=1)
                logps = logps.logsumexp(dim=1) + torch.log(
                    torch.tensor(1.0 / nsamples, device=logps.device)
                )
                loss = (-logps).mean()

        losses.append(loss.item())

        if step % log_every == 0:
            if split == "train":
                print(
                    "Epoch {} Iter {}, {} entropy gap {:.4f} bits (loss {:.3f}, data {:.3f}) {:.5f} lr".format(
                        epoch_num,
                        step,
                        split,
                        loss.item() / np.log(2) - table_bits,
                        loss.item() / np.log(2),
                        table_bits,
                        lr,
                    )
                )
            else:
                print(
                    "Epoch {} Iter {}, {} loss {:.4f} nats / {:.4f} bits".format(
                        epoch_num, step, split, loss.item(), loss.item() / np.log(2)
                    )
                )

        if split == "train":
            opt.zero_grad()
            loss.backward()
            opt.step()

        if verbose:
            print("%s epoch average loss: %f" % (split, np.mean(losses)))
    if return_losses:
        return losses
    return np.mean(losses)

def InvertOrder(order):
    if order is None:
        return None
    # 'order'[i] maps nat_i -> position of nat_i
    # Inverse: position -> natural idx.  This it the 'true' ordering -- it's how
    # heuristic orders are generated + (less crucially) how Transformer works.
    nin = len(order)
    inv_ordering = [None] * nin
    for natural_idx in range(nin):
        inv_ordering[order[natural_idx]] = natural_idx
    return inv_ordering

def MakeMade(scale, cols_to_train, seed, fixed_ordering=None):
    print("MakeMade - START")
    if args.inv_order:
        print("Inverting order!")
        fixed_ordering = InvertOrder(fixed_ordering)

    model = made.MADE(
        nin=len(cols_to_train),
        hidden_sizes=[scale] * args.layers
        if args.layers > 0
        else [512, 256, 512, 128, 1024],
        nout=sum([c.DistributionSize() for c in cols_to_train]),
        input_bins=[c.DistributionSize() for c in cols_to_train],
        input_encoding=args.input_encoding,
        output_encoding=args.output_encoding,
        embed_size=32,
        seed=seed,
        do_direct_io_connections=args.direct_io,
        natural_ordering=False if seed is not None and seed != 0 else True,
        residual_connections=args.residual,
        fixed_ordering=fixed_ordering,
        column_masking=args.column_masking,
    ).to(DEVICE)

    print("MakeMade - END")
    return model

def output_est_cards(est_cards):
    output_est_cards_filepath = './Naru/neurocard_est_cards.txt'
    print(est_cards)
    with open(output_est_cards_filepath, 'w+') as f:
        for est_card in est_cards:
            card_str = ''
            for card in est_card:
                card_str += str(card) + ','
            f.write(card_str[:len(card_str) - 1] + '\n')

class NeuroCard():
    def __init__(self, config):
        self.config = config
        print('NeuroCard config:')
        pprint.pprint(config)
        os.chdir(config['cwd'])
        for k, v in config.items():
            setattr(self, k, v)

        if config['__gpu'] == 0:
            torch.set_num_threads(config['__cpu'])

        self.epoch = 0

        if isinstance(self.join_tables, int):
            # Hack to support training single-model tables.
            sorted_table_names = sorted(
                list(datasets.JoinOrderBenchmark.GetJobLightJoinKeys().keys()))
            self.join_tables = [sorted_table_names[self.join_tables]]

        # Try to make all the runs the same, except for input orderings.
        torch.manual_seed(0)
        np.random.seed(0)

        # Common attributes.
        self.loader = None
        self.join_spec = None
        join_iter_dataset = None
        table_primary_index = None

        # New datasets should be loaded here.
        assert self.dataset in ['imdb']
        if self.dataset == 'imdb':
            print('Training on Join({})'.format(self.join_tables))
            loaded_tables = []
            for t in self.join_tables:
                print('Loading', t)
                table = datasets.LoadImdb(t, use_cols=self.use_cols)
                table.data.info()
                loaded_tables.append(table)
            if len(self.join_tables) > 1:
                join_spec, join_iter_dataset, loader, table = self.MakeSamplerDatasetLoader(
                    loaded_tables)

                self.join_spec = join_spec
                self.train_data = join_iter_dataset
                self.loader = loader

                table_primary_index = [t.name for t in loaded_tables
                                       ].index('title')

                table.cardinality = datasets.JoinOrderBenchmark.GetFullOuterCardinalityOrFail(
                    self.join_tables)
                self.train_data.cardinality = table.cardinality

                print('rows in full join', table.cardinality,
                      'cols in full join', len(table.columns), 'cols:', table)
            else:
                # Train on a single table.
                table = loaded_tables[0]

        if self.dataset != 'imdb' or len(self.join_tables) == 1:
            table.data.info()
            self.train_data = self.MakeTableDataset(table)

        self.table = table
        # Provide true cardinalities in a file or implement an oracle CardEst.
        self.oracle = None
        self.table_bits = 0

        # A fixed ordering?
        self.fixed_ordering = self.MakeOrdering(table)

        model = self.MakeModel(self.table,
                               self.train_data,
                               table_primary_index=table_primary_index)

        # NOTE: ReportModel()'s returned value is the true model size in
        # megabytes containing all all *trainable* parameters.  As impl
        # convenience, the saved ckpts on disk have slightly bigger footprint
        # due to saving non-trainable constants (the masks in each layer) as
        # well.  They can be deterministically reconstructed based on RNG seeds
        # and so should not be counted as model size.
        self.mb = train_utils.ReportModel(model)
        if not isinstance(model, transformer.Transformer):
            print('applying train_utils.weight_init()')
            model.apply(train_utils.weight_init)
        self.model = model

        if self.use_transformer:
            opt = torch.optim.Adam(
                list(model.parameters()),
                2e-4,
                # betas=(0.9, 0.98),  # B in Lingvo; in Trfmr paper.
                betas=(0.9, 0.997),  # A in Lingvo.
                eps=1e-9,
            )
        else:
            if self.optimizer == 'adam':
                opt = torch.optim.Adam(list(model.parameters()), 2e-4)
            else:
                print('Using Adagrad')
                opt = torch.optim.Adagrad(list(model.parameters()), 2e-4)
        print('Optimizer:', opt)
        self.opt = opt

        total_steps = self.epochs * self.max_steps
        if self.lr_scheduler == 'CosineAnnealingLR':
            # Starts decaying to 0 immediately.
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, total_steps)
        elif self.lr_scheduler == 'OneCycleLR':
            # Warms up to max_lr, then decays to ~0.
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=2e-3, total_steps=total_steps)
        elif self.lr_scheduler is not None and self.lr_scheduler.startswith(
                'OneCycleLR-'):
            warmup_percentage = float(self.lr_scheduler.split('-')[-1])
            # Warms up to max_lr, then decays to ~0.
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=2e-3,
                total_steps=total_steps,
                pct_start=warmup_percentage)
        elif self.lr_scheduler is not None and self.lr_scheduler.startswith(
                'wd_'):
            # Warmups and decays.
            splits = self.lr_scheduler.split('_')
            assert len(splits) == 3, splits
            lr, warmup_fraction = float(splits[1]), float(splits[2])
            self.custom_lr_lambda = train_utils.get_cosine_learning_rate_fn(
                total_steps,
                learning_rate=lr,
                min_learning_rate_mult=1e-5,
                constant_fraction=0.,
                warmup_fraction=warmup_fraction)
        else:
            assert self.lr_scheduler is None, self.lr_scheduler

        self.tbx_logger = tune_logger.TBXLogger(self.config, self.logdir)

        if self.checkpoint_to_load:
            self.LoadCheckpoint()

        self.loaded_queries = None
        self.oracle_cards = None
        if self.dataset == 'imdb' and len(self.join_tables) > 1:
            queries_job_format = utils.JobToQuery(self.queries_csv)
            self.loaded_queries, self.oracle_cards = utils.UnpackQueries(
                self.table, queries_job_format)

        if config['__gpu'] == 0:
            print('CUDA not available, using # cpu cores for intra-op:',
                  torch.get_num_threads(), '; inter-op:',
                  torch.get_num_interop_threads())

    def LoadCheckpoint(self):
        all_ckpts = glob.glob(self.checkpoint_to_load)
        msg = 'No ckpt found or use tune.grid_search() for >1 ckpts.'
        assert len(all_ckpts) == 1, msg
        loaded = torch.load(all_ckpts[0])
        try:
            self.model.load_state_dict(loaded)
        except RuntimeError as e:
            # Backward compatibility: renaming.
            def Rename(state_dict):
                new_state_dict = collections.OrderedDict()
                for key, value in state_dict.items():
                    new_key = key
                    if key.startswith('embedding_networks'):
                        new_key = key.replace('embedding_networks',
                                              'embeddings')
                    new_state_dict[new_key] = value
                return new_state_dict

            loaded = Rename(loaded)

            modules = list(self.model.net.children())
            if len(modules) < 2 or type(modules[-2]) != nn.ReLU:
                raise e
            # Try to load checkpoints created prior to a 7/28/20 fix where
            # there's an activation missing.
            print('Try loading without ReLU before output layer.')
            modules.pop(-2)
            self.model.net = nn.Sequential(*modules)
            self.model.load_state_dict(loaded)

        print('Loaded ckpt from', all_ckpts[0])

    def MakeTableDataset(self, table):
        train_data = common.TableDataset(table)
        if self.factorize:
            train_data = common.FactorizedTable(
                train_data, word_size_bits=self.word_size_bits)
        return train_data

    def MakeSamplerDatasetLoader(self, loaded_tables):
        assert self.sampler in ['fair_sampler',
                                'factorized_sampler'], self.sampler
        join_spec = join_utils.get_join_spec(self.__dict__)
        klass = factorized_sampler.FactorizedSamplerIterDataset
        join_iter_dataset = klass(
            loaded_tables,
            join_spec,
            sample_batch_size=self.sampler_batch_size,
            disambiguate_column_names=True,
            # Only initialize the sampler if training.
            initialize_sampler=self.checkpoint_to_load is None,
            save_samples=self._save_samples,
            load_samples=self._load_samples)

        table = common.ConcatTables(loaded_tables,
                                    self.join_keys,
                                    sample_from_join_dataset=join_iter_dataset)

        if self.factorize:
            join_iter_dataset = common.FactorizedSampleFromJoinIterDataset(
                join_iter_dataset,
                base_table=table,
                factorize_blacklist=self.dmol_cols if self.num_dmol else
                self.factorize_blacklist if self.factorize_blacklist else [],
                word_size_bits=self.word_size_bits,
                factorize_fanouts=self.factorize_fanouts)

        loader = data.DataLoader(join_iter_dataset,
                                 batch_size=self.bs,
                                 num_workers=self.loader_workers,
                                 worker_init_fn=lambda worker_id: np.random.
                                 seed(np.random.get_state()[1][0] + worker_id),
                                 pin_memory=True)
        return join_spec, join_iter_dataset, loader, table

    def MakeOrdering(self, table):
        fixed_ordering = None
        if self.dataset != 'imdb' and self.special_orders <= 1:
            fixed_ordering = list(range(len(table.columns)))

        if self.order is not None:
            print('Using passed-in order:', self.order)
            fixed_ordering = self.order

        if self.order_seed is not None:
            if self.order_seed == 'reverse':
                fixed_ordering = fixed_ordering[::-1]
            else:
                rng = np.random.RandomState(self.order_seed)
                rng.shuffle(fixed_ordering)
            print('Using generated order:', fixed_ordering)
        return fixed_ordering

    def MakeModel(self, table, train_data, table_primary_index=None):
        cols_to_train = table.columns
        if self.factorize:
            cols_to_train = train_data.columns

        fixed_ordering = self.MakeOrdering(table)

        table_num_columns = table_column_types = table_indexes = None
        if isinstance(train_data, (common.SamplerBasedIterDataset,
                                   common.FactorizedSampleFromJoinIterDataset)):
            table_num_columns = train_data.table_num_columns
            table_column_types = train_data.combined_columns_types
            table_indexes = train_data.table_indexes
            print('table_num_columns', table_num_columns)
            print('table_column_types', table_column_types)
            print('table_indexes', table_indexes)
            print('table_primary_index', table_primary_index)

        if self.use_transformer:
            args = {
                'num_blocks': 4,
                'd_ff': 128,
                'd_model': 32,
                'num_heads': 4,
                'd_ff': 64,
                'd_model': 16,
                'num_heads': 2,
                'nin': len(cols_to_train),
                'input_bins': [c.distribution_size for c in cols_to_train],
                'use_positional_embs': False,
                'activation': 'gelu',
                'fixed_ordering': self.fixed_ordering,
                'dropout': self.dropout,
                'per_row_dropout': self.per_row_dropout,
                'seed': None,
                'join_args': {
                    'num_joined_tables': len(self.join_tables),
                    'table_dropout': self.table_dropout,
                    'table_num_columns': table_num_columns,
                    'table_column_types': table_column_types,
                    'table_indexes': table_indexes,
                    'table_primary_index': table_primary_index,
                }
            }
            args.update(self.transformer_args)
            model = transformer.Transformer(**args).to(train_utils.get_device())
        else:
            model = MakeMade(
                table=table,
                scale=self.fc_hiddens,
                layers=self.layers,
                cols_to_train=cols_to_train,
                seed=self.seed,
                factor_table=train_data if self.factorize else None,
                fixed_ordering=fixed_ordering,
                special_orders=self.special_orders,
                order_content_only=self.order_content_only,
                order_indicators_at_front=self.order_indicators_at_front,
                inv_order=True,
                residual=self.residual,
                direct_io=self.direct_io,
                input_encoding=self.input_encoding,
                output_encoding=self.output_encoding,
                embed_size=self.embed_size,
                dropout=self.dropout,
                per_row_dropout=self.per_row_dropout,
                grouped_dropout=self.grouped_dropout
                if self.factorize else False,
                fixed_dropout_ratio=self.fixed_dropout_ratio,
                input_no_emb_if_leq=self.input_no_emb_if_leq,
                embs_tied=self.embs_tied,
                resmade_drop_prob=self.resmade_drop_prob,
                # DMoL:
                num_dmol=self.num_dmol,
                scale_input=self.scale_input if self.num_dmol else False,
                dmol_cols=self.dmol_cols if self.num_dmol else [],
                # Join specific:
                num_joined_tables=len(self.join_tables),
                table_dropout=self.table_dropout,
                table_num_columns=table_num_columns,
                table_column_types=table_column_types,
                table_indexes=table_indexes,
                table_primary_index=table_primary_index,
            )
        return model

    def MakeProgressiveSamplers(self,
                                model,
                                train_data,
                                do_fanout_scaling=False):
        estimators = []
        dropout = self.dropout or self.per_row_dropout
        for n in self.eval_psamples:
            if self.factorize:
                estimators.append(
                    estimators_lib.FactorizedProgressiveSampling(
                        model,
                        train_data,
                        n,
                        self.join_spec,
                        device=train_utils.get_device(),
                        shortcircuit=dropout,
                        do_fanout_scaling=do_fanout_scaling))
            else:
                estimators.append(
                    estimators_lib.ProgressiveSampling(
                        model,
                        train_data,
                        n,
                        self.join_spec,
                        device=train_utils.get_device(),
                        shortcircuit=dropout,
                        do_fanout_scaling=do_fanout_scaling))
        return estimators

    def _simple_save(self):
        save_dir=""
        path = os.path.join(
            "./Naru/checkpoints", 'model-{}-{}.h5'.format(self.epoch,
                                                   '-'.join(self.join_tables)))
        torch.save(self.model.state_dict(), path)
        return path

    def train(self):
        if self.checkpoint_to_load or self.eval_join_sampling:
            self.model.model_bits = 0
            results = self.evaluate(self.num_eval_queries_at_checkpoint_load,
                                               done=True)
            self._maybe_check_asserts(results, returns=None)
            return {
                'epoch': 0,
                'done': True,
                'results': results,
            }

        for _ in range(min(self.epochs - self.epoch,
                           self.epochs_per_iteration)):
            mean_epoch_train_loss = RunEpoch(
                split='train',
                model=self.model,
                opt=self.opt,
                upto=self.max_steps if self.dataset == 'imdb' else None,
                train_data=self.train_data,
                val_data=self.train_data,
                batch_size=self.bs,
                epoch_num=self.epoch,
                log_every=100,
                table_bits=self.table_bits)
            self.epoch += 1
        self.model.model_bits = mean_epoch_train_loss / np.log(2)

        if self.checkpoint_every_epoch:
            self._simple_save()

        done = self.epoch >= self.epochs
        results = self.evaluate(
            max(self.num_eval_queries_at_end,
                self.num_eval_queries_per_iteration)
            if done else self.num_eval_queries_per_iteration, done)

        returns = {
            'epochs': self.epoch,
            'done': done,
            'avg_loss': self.model.model_bits - self.table_bits,
            'train_bits': self.model.model_bits,
            'train_bit_gap': self.model.model_bits - self.table_bits,
            'results': results,
        }

        if self.compute_test_loss:
            returns['test_bits'] = np.mean(
                RunEpoch(
                    'test',
                    self.model,
                    opt=None,
                    train_data=self.train_data,
                    val_data=self.train_data,
                    batch_size=1024,
                    upto=None if self.dataset != 'imdb' else 20,
                    log_every=200,
                    table_bits=self.table_bits,
                    return_losses=True,
                )) / np.log(2)
            self.model.model_bits = returns['test_bits']
            print('Test bits:', returns['test_bits'])

        if done:
            self._maybe_check_asserts(results, returns)

        return returns

    def _maybe_check_asserts(self, results, returns):
        if self.asserts:
            # asserts = {key: val, ...} where key either exists in "results"
            # (returned by evaluate()) or "returns", both defined above.
            error = False
            message = []
            for key, max_val in self.asserts.items():
                if key in results:
                    if results[key] >= max_val:
                        error = True
                        message.append(str((key, results[key], max_val)))
                elif returns[key] >= max_val:
                    error = True
                    message.append(str((key, returns[key], max_val)))
            assert not error, '\n'.join(message)

    def save(self, tmp_checkpoint_dir):
        if self.checkpoint_to_load or not self.save_checkpoint_at_end:
            return {}

        # NOTE: see comment at ReportModel() call site about model size.
        if self.fixed_ordering is None:
            if self.seed is not None:
                PATH = '{}-{:.1f}MB-model{:.3f}-{}-{}epochs-seed{}.pt'.format(
                    self.dataset, self.mb, self.model.model_bits,
                    self.model.name(), self.epoch, self.seed)
            else:
                PATH = '{}-{:.1f}MB-model{:.3f}-{}-{}epochs-seed{}-{}.pt'.format(
                    self.dataset, self.mb, self.model.model_bits,
                    self.model.name(), self.epoch, self.seed, time.time())
        else:
            PATH = '{}-{:.1f}MB-model{:.3f}-{}-{}epochs-seed{}-order{}.pt'.format(
                self.dataset, self.mb, self.model.model_bits, self.model.name(),
                self.epoch, self.seed,
                str(self.order_seed) if self.order_seed is not None else
                '_'.join(map(str, self.fixed_ordering))[:60])

        if self.dataset == 'imdb':
            tuples_seen = self.bs * self.max_steps * self.epochs
            PATH = PATH.replace(
                '-seed', '-{}tups-seed'.format(utils.HumanFormat(tuples_seen)))

            if len(self.join_tables) == 1:
                PATH = PATH.replace('imdb',
                                    'indep-{}'.format(self.join_tables[0]))
        PATH=os.path.join(tmp_checkpoint_dir, PATH)
        torch.save(self.model.state_dict(), PATH)
        print('Saved to:', PATH)
        return {'path': PATH}

    def stop(self):
        self.tbx_logger.flush()
        self.tbx_logger.close()

    def _log_result(self, results):
        psamples = {}
        # When we run > 1 epoch in one tune "iter", we want TensorBoard x-axis
        # to show our real epoch numbers.
        results['iterations_since_restore'] = results[
            'training_iteration'] = self.epoch
        for k, v in results['results'].items():
            if 'psample' in k:
                psamples[k] = v
        self.tbx_logger.on_result(results)
        self.tbx_logger._file_writer.add_custom_scalars_multilinechart(
            map(lambda s: 'ray/tune/results/{}'.format(s), psamples.keys()),
            title='psample')

    def ErrorMetric(self, est_card, card):
        if card == 0 and est_card != 0:
            return est_card
        if card != 0 and est_card == 0:
            return card
        if card == 0 and est_card == 0:
            return 1.0
        return max(est_card / card, card / est_card)

    def Query(self,
              estimators,
              oracle_card=None,
              query=None,
              table=None,
              oracle_est=None):
        est_cards = []
        assert query is not None
        cols, ops, vals = query
        card = oracle_est.Query(cols, ops,
                                vals) if oracle_card is None else oracle_card
        print('Q(', end='')
        for c, o, v in zip(cols, ops, vals):
            print('{} {} {}, '.format(c.name, o, str(v)), end='')
        print('): ', end='')
        print('\n  actual {} ({:.3f}%) '.format(card,
                                                card / table.cardinality * 100),
              end='')
        for est in estimators:
            est_card = est.Query(cols, ops, vals)
            est_cards.append(est_card)
            err = self.ErrorMetric(est_card, card)
            est.AddError(err, est_card, card)
            print('{} {} (err={:.3f}) '.format(str(est), est_card, err), end='')
        print()
        return est_cards

    def evaluate(self, num_queries, estimators=None):
        model = self.model

        model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        est_cards = []
        results = {}
        if num_queries:
            if estimators is None:
                estimators = self.MakeProgressiveSamplers(
                    model,
                    self.train_data if self.factorize else self.table,
                    do_fanout_scaling=(self.dataset == 'imdb'))
                if self.eval_join_sampling:  # None or an int.
                    estimators = [
                        estimators_lib.JoinSampling(self.train_data, self.table,
                                                    self.eval_join_sampling)
                    ]

            assert self.loaded_queries is not None
            num_queries = min(len(self.loaded_queries), num_queries)
            for i in range(num_queries):
                print('Query {}:'.format(i), end=' ')
                query = self.loaded_queries[i]
                est_card = self.Query(estimators,
                                      oracle_card=None if self.oracle_cards is None else
                                      self.oracle_cards[i],
                                      query=query,
                                      table=self.table,
                                      oracle_est=self.oracle)
                est_cards.append(est_card)
                if i % 100 == 0:
                    for est in estimators:
                        est.report()

            for est in estimators:
                results[str(est) + '_max'] = np.max(est.errs)
                results[str(est) + '_p99'] = np.quantile(est.errs, 0.99)
                results[str(est) + '_p95'] = np.quantile(est.errs, 0.95)
                results[str(est) + '_median'] = np.median(est.errs)
                est.report()

                series = pd.Series(est.query_dur_ms)
                print(series.describe())
                series.to_csv(str(est) + '.csv', index=False, header=False)

        output_est_cards(est_cards)
        return results

if __name__ == "__main__":
    config = experiments.EXPERIMENT_CONFIGS["job-light"]
    model_train=NeuroCard(config)
    returns=model_train.train()
    model_train.save("./Naru/checkpoints")
