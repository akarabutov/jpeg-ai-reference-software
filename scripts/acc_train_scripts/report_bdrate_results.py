# The copyright in this software is being made available under the BSD
# License, included below. This software may be subject to other third party
# and contributor rights, including patent rights, and no such rights are
# granted under this license.
#
# Copyright (c) 2010-2022, ITU/ISO/IEC
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
# be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.

#!/usr/bin/env python
import os
import shutil
import tempfile
from typing import List, Tuple

# from bjontegaard.functions import bd_rate #TODO use pip package once it is available on cluster
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from bjontegaard.functions import bd_rate


class Config:
    stages = [
        'MSE_FixedRate_64', 'Mixed_FixedRate_36', 'Mixed_FixedRate_OnlyDec_20',
        'MSE_VariableRate_12'
    ]
    summary_column_names = [
        'name', 'bpp', 'msssim_Torch', 'msssim_iqa', 'psnrY', 'psnrU', 'psnrV', 'vif', 'fsim',
        'nlpd', 'iw_ssim', 'vmaf', 'psnrHVS', 'dummy1', 'dummy2', 'dummy3', 'dummy4', 'dummy5'
    ]
    all_metrics = [
        'msssim_Torch',
        'msssim_iqa',
        'psnrY',
        'psnrU',
        'psnrV',
        'vif',
        'fsim',
        'nlpd',
        'iw_ssim',
        'vmaf',
        'psnrHVS',
    ]


class BDRateReporter:
    def __init__(self, train_url: str) -> None:
        self.train_url = train_url

        # this will be called periodically by main thread, but we do not want to
        # redo tensorboard plot everytime. For keeping track which epochs/stages
        # were already handled in summary writer
        self.processed_by_summary_writer = dict()
        stages = [
            'MSE_FixedRate_64', 'Mixed_FixedRate_36', 'Mixed_FixedRate_OnlyDec_20',
            'MSE_VariableRate_12'
        ]
        for stage in stages:
            self.processed_by_summary_writer[stage] = set()
        self.summary_writer = self.init_summary_writer(self.train_url)

    def read_anchor_data(self, anchor_file: str) -> pd.DataFrame:
        """Read anchor data for bd rate comparison. Format: Separated by tab.

        Args:
            anchor_file (str): path to file with anchor data.

        Returns:
            pd.DataFrame: Dataframe holding the parsed anchor data
        """

        anchor_vvc_data = pd.read_csv(anchor_file, sep='\t', names=Config.summary_column_names)
        anchor_vvc_data.drop(columns=['dummy1', 'dummy2', 'dummy3', 'dummy4', 'dummy5'],
                             inplace=True)

        image_and_beta = anchor_vvc_data.name.str.extract('VVC_(.*)_(\d+).png')  # noqa: W605
        anchor_vvc_data['image'] = image_and_beta.loc[:, 0]
        anchor_vvc_data['beta'] = image_and_beta.loc[:, 1].astype('int')

        return anchor_vvc_data

    def add_summary(self, results_dir: str, epoch: int, stage: str, summaries_list: List) -> None:
        """Adds summary data for a given stage, epoch to list of collected summary data.

        Args:
            results_dir (str): direcotry of results
            epoch (int): epoch of results
            stage (str): stage of results
            summaries_list (List): list to which data will be added
        """
        summary_file = os.path.join(results_dir, f'epoch{epoch}', 'summary.txt')
        if os.path.isfile(summary_file):
            summary_data = pd.read_csv(summary_file, sep='\t', names=Config.summary_column_names)
            summary_data['epoch'] = epoch
            summary_data['stage'] = stage

            # remove lines that we're missing in summary ("NO metrics")
            summary_data = summary_data[summary_data.name.str.contains('VM')]

            # drop unneeded data and extract image and beta
            summary_data.drop(columns=[
                'dummy1',
                'dummy2',
                'dummy3',
                'dummy4',
            ], inplace=True)
            image_and_beta = summary_data.name.str.extract('VM_(.*)_(\d+).png')  # noqa: W605
            summary_data['image'] = image_and_beta.loc[:, 0]
            summary_data['beta'] = image_and_beta.loc[:, 1].astype('int')

            # make sure types for fields are correct
            summary_data.bpp = summary_data.bpp.astype('float')
            for metric in Config.all_metrics:
                summary_data[metric] = summary_data[metric].astype('float', errors='ignore')

            summaries_list.append(summary_data)

    def get_collected_summaries(self, train_out_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Search train output dir for summary files and parse them.

        Args:
            train_out_dir (str): path of train output directory.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Collected data for all stages and epochs, collected
            data for best of each stage.
        """
        collected_summary_data = []
        best_of_stages_data = []

        for stage in Config.stages:
            stage_dir = os.path.join(train_out_dir, f'{stage}')
            if os.path.isdir(stage_dir):
                results_dir = os.path.join(stage_dir, 'results')
                if os.path.isdir(results_dir):
                    available_epochs = [
                        epoch_dir for epoch_dir in os.listdir(results_dir)
                        if os.path.isdir(os.path.join(results_dir, epoch_dir))
                    ]

                    available_epochs2 = []
                    add_best_epoch = False
                    for epoch_name in available_epochs:
                        if 'epoch' in epoch_name:
                            if 'best' in epoch_name:
                                add_best_epoch = True
                            else:
                                available_epochs2.append(int(epoch_name.split('epoch')[1]))

                    available_epochs = sorted(available_epochs2)

                    for epoch in available_epochs:
                        self.add_summary(results_dir, epoch, stage, collected_summary_data)

                    if add_best_epoch:
                        self.add_summary(results_dir, 'best', stage, best_of_stages_data)

        if collected_summary_data:
            collected_summary_data = pd.concat(collected_summary_data)
        else:
            collected_summary_data = pd.DataFrame()
        if best_of_stages_data:
            best_of_stages_data = pd.concat(best_of_stages_data)
        else:
            best_of_stages_data = pd.DataFrame()

        return collected_summary_data, best_of_stages_data

    def calc_bd_rates(self, anchor_vvc_data: pd.DataFrame,
                      collected_summary_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """For each summary data compute bd rate results relative to the anchor data.

        Args:
            anchor_vvc_data (pd.DataFrame): data frame with anchor data
            collected_summary_data (pd.DataFrame): data frame with the collected summary data

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: data frame with computed BD-Rates, data frame indicating where erros occured in metric calculation.
        """
        anchor_vvc_data_beta_6_to_100 = anchor_vvc_data[(anchor_vvc_data.beta >= 6)
                                                       & (anchor_vvc_data.beta <= 100)]
        anchor_vvc_data_pv = anchor_vvc_data_beta_6_to_100.pivot_table(index=['image', 'beta'])
        collected_summary_data_beta_6_to_100 = collected_summary_data[
            (collected_summary_data.beta >= 6) & (collected_summary_data.beta <= 100)]
        collected_summary_data_pv = collected_summary_data_beta_6_to_100.pivot_table(
            index=['stage', 'epoch', 'image', 'beta'])

        bdrates_df = []
        errors_df = []
        stages = collected_summary_data.stage.unique()
        images = collected_summary_data.image.unique()

        for stage in stages:
            stage_df = collected_summary_data_pv.xs(stage)
            epochs_in_stage = stage_df.reset_index().epoch.unique()
            for epoch in epochs_in_stage:
                epoch_df = stage_df.xs(epoch)
                for image in images:
                    image_test_df = epoch_df.xs(image)
                    image_anchor_df = anchor_vvc_data_pv.xs(image)

                    metric_results = {}
                    errors = {}
                    for metric in image_test_df.drop(columns='bpp').columns:
                        # if metric in ['psnrU', 'psnrV']:
                        #     continue
                        try:
                            if metric == 'nlpd':
                                metric_results[metric] = [
                                    bd_rate(image_anchor_df.bpp.to_numpy(),
                                            1 - image_anchor_df[metric].to_numpy(),
                                            image_test_df.bpp.to_numpy(),
                                            1 - image_test_df[metric].to_numpy(),
                                            'pchip',
                                            require_matching_points=True)
                                ]
                            else:
                                metric_results[metric] = [
                                    bd_rate(image_anchor_df.bpp.to_numpy(),
                                            image_anchor_df[metric].to_numpy(),
                                            image_test_df.bpp.to_numpy(),
                                            image_test_df[metric].to_numpy(),
                                            'pchip',
                                            require_matching_points=True)
                                ]
                            errors[metric] = [0]
                        except (ValueError, AssertionError):
                            # print(f'Error in metrics at {stage}, {epoch}, {image}, {metric}') # TODO: remove or do with logger so its not always shown
                            metric_results[metric] = [
                                1e6
                            ]  # dummy value. can not use None/NaN (see `replace_errors_with_NA` below)
                            errors[metric] = [1]
                            continue

                    metric_results['image'] = [image]
                    metric_results['stage'] = [stage]
                    metric_results['epoch'] = [epoch]
                    errors['image'] = [image]
                    errors['stage'] = [stage]
                    errors['epoch'] = [epoch]
                    bdrates_df.append(pd.DataFrame(metric_results))
                    errors_df.append(pd.DataFrame(errors))

        bdrates_df = pd.concat(bdrates_df)
        errors_df = pd.concat(errors_df)
        return bdrates_df, errors_df

    def get_metric_averages(self, bdrates_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate average of metrics for per stage and epoch in data frame and add it to data
        frame as new column.

        Args:
            bdrates_df (pd.DataFrame): data frame to augment with calculated average metrics.

        Returns:
            pd.DataFrame: the augmented data frame.
        """

        bdrates_pv = bdrates_df.pivot_table(
            index=['stage', 'epoch'],
            values=['msssim_Torch', 'vif', 'fsim', 'nlpd', 'iw_ssim', 'vmaf', 'psnrHVS'],
            aggfunc={
                'fsim': np.mean,  # noqa: E122
                'iw_ssim': np.mean,
                'msssim_Torch': np.mean,
                'nlpd': np.mean,
                'psnrHVS': np.mean,
                'vif': np.mean,
                'vmaf': np.mean
            })

        stages = bdrates_pv.reset_index().stage.unique()

        for stage in stages:
            stage_df = bdrates_pv.xs(stage)
            epochs_in_stage = stage_df.reset_index().epoch.unique()
            for epoch in epochs_in_stage:
                average = bdrates_pv.xs((stage, epoch)).mean()
                bdrates_pv.loc[(stage, epoch), 'Average'] = average

        bdrates_pv = bdrates_pv[[
            'Average',
            'msssim_Torch',
            'vif',
            'fsim',
            'nlpd',
            'iw_ssim',
            'vmaf',
            'psnrHVS',
        ]]

        return bdrates_pv

    def collect_metric_errors(self, errors_df: pd.DataFrame) -> pd.DataFrame:
        """Determin which average metrics values are erroneous based on error data frame.

        Args:
            errors_df (pd.DataFrame): data frame holding errors per epoch/stage

        Returns:
            pd.DataFrame: augmented data frame also indicating which average values are erroneous
        """

        errors_pv = errors_df.pivot_table(
            index=['stage', 'epoch'],
            values=['msssim_Torch', 'vif', 'fsim', 'nlpd', 'iw_ssim', 'vmaf', 'psnrHVS'],
            aggfunc=np.max)

        stages = errors_pv.reset_index().stage.unique()

        for stage in stages:
            stage_df = errors_pv.xs(stage)
            epochs_in_stage = stage_df.reset_index().epoch.unique()
            for epoch in epochs_in_stage:
                error_in_average = errors_pv.xs((stage, epoch)).max()
                errors_pv.loc[(stage, epoch), 'Average'] = error_in_average

        errors_pv.Average = errors_pv.Average.astype('int')
        errors_pv = errors_pv[[
            'Average',
            'msssim_Torch',
            'vif',
            'fsim',
            'nlpd',
            'iw_ssim',
            'vmaf',
            'psnrHVS',
        ]]

        return errors_pv

    def replace_errors_with_NA(self, bdrates_pv: pd.DataFrame,
                               errors_pv: pd.DataFrame) -> pd.DataFrame:
        """We used pandas.pivot to process data so far. However, pivot would automatically remove NA
        (which we get when there are errors, inf, nan, etc.). Thus the hack of storing error locations
        and replacing metrics at error locations with dummy values.
        This function merges both our obtained metric and error data frames. Putting NA in places where
        we have errors.

        Args:
            bdrates_pv (pd.DataFrame): dataframe with metrics, after pivot
            errors_pv (pd.DataFrame): dataframe with errors, after pivot

        Returns:
            pd.DataFrame: metrics data frame, after pivot, error locations set to `pd.NA`
        """

        overview_metrics = [
            'Average',
            'msssim_Torch',
            'vif',
            'fsim',
            'nlpd',
            'iw_ssim',
            'vmaf',
            'psnrHVS',
        ]

        bdrates_out = bdrates_pv.copy().reset_index()
        errors_out = errors_pv.copy().reset_index()

        stages = bdrates_pv.reset_index().stage.unique()

        for stage in stages:
            stage_df = bdrates_pv.xs(stage)
            epochs_in_stage = stage_df.reset_index().epoch.unique()
            for epoch in epochs_in_stage:
                row_index = bdrates_out[(bdrates_out.stage == stage)
                                        & (bdrates_out.epoch == epoch)].index
                for metric in overview_metrics:
                    if errors_out.loc[row_index, metric].item():
                        bdrates_out.loc[row_index, metric] = None

        bdrates_out_pv = bdrates_out.set_index(
            ['stage',
             'epoch'])  # pivot again, using set index since we do not need to aggregate anymore
        return bdrates_out_pv

    def plot_per_epoch_bdrates(self, bdrates_pv: pd.DataFrame, bdrates_best_pv: pd.DataFrame):
        """Plot average bd rate metric progress over stages and epochs. Return generated figure.

        Args:
            bdrates_pv (pd.DataFrame): data frame holding average BD-rate data per stage/epoch
            bdrates_best_pv (pd.DataFrame): data frame holding data for best of stage BD-rate

        Returns:
            _type_: Figure showing the plot of average BD-rate progession.
        """
        if bdrates_pv.empty:
            return None

        fig = plt.figure()

        stages = bdrates_pv.reset_index().stage.unique()
        cmap = plt.cm.jet(np.linspace(0, 1, len(stages)))

        num_epochs_processed_stages = 0
        for stage, color in zip(stages, cmap):
            stage_df = bdrates_pv.xs(stage)
            epochs_in_stage = stage_df.reset_index().epoch.unique()
            overall_epochs = epochs_in_stage + num_epochs_processed_stages

            # omit plotting first epoch of first stage. still to random, huge error will
            # hide changes for later epochs
            if stage == 'MSE_FixedRate_64':
                overall_epochs = overall_epochs[1:]
                average = stage_df.Average[1:]
            else:
                overall_epochs = overall_epochs
                average = stage_df.Average

            if average.empty:
                continue  # not enough data yet

            plt.plot(overall_epochs,
                     average,
                     label=stage,
                     marker='o',
                     linestyle='solid',
                     color=color)

            # need so that line of next stage is continuing previous one
            # TODO: use time instead, need to read it from log
            num_epochs_processed_stages += epochs_in_stage.max()

        for stage, color in zip(stages, cmap):
            if not bdrates_best_pv.empty:
                stage_best_df = bdrates_best_pv.xs(stage)
                if not stage_best_df.empty:
                    best = stage_best_df.Average
                    plt.hlines(best,
                               0,
                               overall_epochs[-1],
                               label=f'{stage} best',
                               linestyle='dashed',
                               color=color)

        plt.grid()
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Average BD-Rate')

        return fig

    def init_summary_writer(self, train_url: str) -> SummaryWriter:
        """Initalize tensorbaord summary writer. Add overlay for showing BD-Rate results for
        different stages in same figure.

        Args:
            train_url (str): training output directory path

        Returns:
            SummaryWriter: Initialized summary writer instance.
        """

        self.tensor_board_out_dir = os.path.join(train_url, 'tensorboard')
        os.makedirs(self.tensor_board_out_dir, exist_ok=True)
        # tmp dir is workaround need for cluster file system, not supported by summary writer
        self.summary_writer_tmp_dir = tempfile.TemporaryDirectory()
        summary_writer = SummaryWriter(self.summary_writer_tmp_dir.name)

        # add a plot showing train and validation loss together
        stages = [
            'MSE_FixedRate_64', 'Mixed_FixedRate_36', 'Mixed_FixedRate_OnlyDec_20',
            'MSE_VariableRate_12'
        ]
        layout = {
            'Test BD-Rate per stage': {
                'BD-Rate': ['Multiline', [f'bd_rate_{stage}/train' for stage in stages]],
            },
        }
        summary_writer.add_custom_scalars(layout)

        return summary_writer

    def plot_per_epoch_bdrates_tensorboard(self, bdrates_pv: pd.DataFrame) -> None:
        """Plot average BD-Rates using tensorboard

        Args:
            bdrates_pv (pd.DataFrame): data frame holding average BD-rates per stage/epoch
        """

        if bdrates_pv.empty:
            return

        stages = bdrates_pv.reset_index().stage.unique()

        for stage in stages:
            stage_df = bdrates_pv.xs(stage)
            epochs_in_stage = stage_df.reset_index().epoch.unique()
            for epoch in epochs_in_stage:

                if epoch in self.processed_by_summary_writer[stage]:
                    continue
                else:
                    self.processed_by_summary_writer[stage].add(epoch)
                    average = stage_df.xs(epoch).Average
                    self.summary_writer.add_scalar(f'bd_rate_{stage}/train', average, epoch)

    def copy_and_overwrite_tensorboard_output(self):
        tmp_dir_tb_files = os.listdir(self.summary_writer_tmp_dir.name)

        # only files, not dirs
        tmp_dir_tb_files = [
            file for file in tmp_dir_tb_files
            if os.path.isfile(os.path.join(self.summary_writer_tmp_dir.name, file))
        ]

        # only tensor board files
        tmp_dir_tb_files = [file for file in tmp_dir_tb_files if 'events.out.tfevents' in file]

        # copy and overwrite
        for file in tmp_dir_tb_files:
            shutil.copyfile(os.path.join(self.summary_writer_tmp_dir.name, file),
                            os.path.join(self.tensor_board_out_dir, file))

    def report_bdrate_results(self) -> None:
        """Read anchor and summary data, compute BD-rates and report them as html tables and
        also as generated figure (png) and via tensorboard output directory.
        """
        anchor_vvc_data = self.read_anchor_data('scripts/acc_train_scripts/anchor_VVC.txt')

        collected_summary_data, best_of_stages_data = self.get_collected_summaries(self.train_url)

        if not collected_summary_data.empty:
            images = collected_summary_data.image.unique()
            assert (anchor_vvc_data.image.unique() == images).all()

            bdrates, errors = self.calc_bd_rates(anchor_vvc_data, collected_summary_data)
            per_image_bdrates_out = bdrates.pivot_table(index=['image', 'stage', 'epoch']).round(2)
            per_image_bdrates_out.to_html(
                os.path.join(self.train_url, 'TrainingProgressPerImageData.html'))
            per_image_bdrates_out.to_excel(
                os.path.join(self.train_url, 'TrainingProgressPerImageData.xlsx'))

            # drop SCC images
            # bdrates_wo_SCC = self.drop_SCC_images(bdrates)
            # errors_wo_SCC = self.drop_SCC_images(errors)
 
            bdrates_pv = self.get_metric_averages(bdrates)
            errors_pv = self.collect_metric_errors(errors)

            bdrates_pv = self.replace_errors_with_NA(bdrates_pv, errors_pv)

            bdrates_pv.round(2).to_html(os.path.join(self.train_url, 'TrainingProgress.html'))
            bdrates_pv.round(2).to_excel(os.path.join(self.train_url, 'TrainingProgress.xlsx'))
        else:
            bdrates_pv = pd.DataFrame()

        if not best_of_stages_data.empty:
            bdrates_best, errors_best = self.calc_bd_rates(anchor_vvc_data, best_of_stages_data)
            # bdrates_best_wo_SCC = self.drop_SCC_images(bdrates_best)
            # errors_best_wo_SCC = self.drop_SCC_images(errors_best)

            bdrates_best_pv = self.get_metric_averages(bdrates_best)
            # errors_best_pv = self.collect_metric_errors(errors_best_wo_SCC)

            bdrates_best_pv.round(2).to_html(os.path.join(self.train_url, 'StagesBest.html'))
            bdrates_best_pv.round(2).to_excel(os.path.join(self.train_url, 'StagesBest.xlsx'))
        else:
            bdrates_best_pv = pd.DataFrame()

        fig = self.plot_per_epoch_bdrates(bdrates_pv, bdrates_best_pv)
        if fig:
            fig.savefig(os.path.join(self.train_url, 'TrainingProgress.png'))
        self.plot_per_epoch_bdrates_tensorboard(bdrates_pv)

        # tmp dir is workaround need for cluster file system, not supported by summary writer
        # now copy data to actual output directory
        self.copy_and_overwrite_tensorboard_output()

    def close_summary_writer(self) -> None:
        """Close summary writer instance
        """
        self.summary_writer.close()
        self.copy_and_overwrite_tensorboard_output()
