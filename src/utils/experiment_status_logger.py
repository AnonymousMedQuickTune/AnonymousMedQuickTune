#!/usr/bin/env python3
"""
Experiment Status Logger for tracking NePS training progress and evaluation status.
Uses separate files for NePS and evaluation status to prevent data loss.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class ExperimentStatusLogger:
    """
    Logger for tracking experiment status using separate files for NePS, QuickTune, and evaluation.
    """
    
    def __init__(self, experiment_base_dir: str, experiment_type: str = "neps"):
        """
        Initialize the logger for a specific experiment.
        
        Args:
            experiment_base_dir (str): Base directory of the experiment
            experiment_type (str): Type of experiment ("neps" or "quicktune")
        """
        self.experiment_base_dir = Path(experiment_base_dir)
        self.experiment_type = experiment_type
        self.status_dir = self.experiment_base_dir / "experiment_status"
        self.status_dir.mkdir(exist_ok=True)
        
        # Choose status file based on experiment type
        if experiment_type == "quicktune":
            self.main_status_file = self.status_dir / "medquicktune_status.txt"
        else:  # neps
            self.main_status_file = self.status_dir / "neps_status.txt"
            
        self.evaluation_status_file = self.status_dir / "evaluation_status.txt"
        
        # Initialize status data - but don't overwrite existing data
        self.main_status = self._load_main_status()
        self.evaluation_status = self._load_evaluation_status()
        
        # Preserve existing evaluation data if available
        if self.evaluation_status_file.exists():
            try:
                with open(self.evaluation_status_file, 'r') as f:
                    content = f.read()
                    existing_status = self._parse_evaluation_status(content)
                    
                    if existing_status['evaluation_count'] > 0:
                        self.evaluation_status['evaluation_count'] = existing_status['evaluation_count']
                    if existing_status['evaluation_history']:
                        self.evaluation_status['evaluation_history'] = existing_status['evaluation_history']
            except Exception:
                pass
        

    
    def _load_main_status(self) -> Dict:
        """Load main status from file or create new if not exists."""
        if self.main_status_file.exists():
            try:
                with open(self.main_status_file, 'r') as f:
                    content = f.read()
                    if self.experiment_type == "quicktune":
                        parsed_status = self._parse_quicktune_status(content)
                    else:  # neps
                        parsed_status = self._parse_neps_status(content)
            
                    return parsed_status
            except Exception:
                pass
        
        # Create new status structure
        return {
            'started': None,
            'finished': None,
            'last_updated': None,
            'total_outer_folds': 0,
            'completed_outer_folds': 0,
            'outer_folds_progress': {}
        }
    
    def _load_evaluation_status(self) -> Dict:
        """Load evaluation status from file or create new if not exists."""
        if self.evaluation_status_file.exists():
            try:
                with open(self.evaluation_status_file, 'r') as f:
                    content = f.read()
                    parsed_status = self._parse_evaluation_status(content)
                    
                    # If we successfully parsed existing data, use it
                    if parsed_status['started'] is not None or parsed_status['evaluation_count'] > 0:
                        return parsed_status
            except Exception:
                pass
        
        # Create new evaluation status structure
        return {
            'started': None,
            'finished': None,
            'last_updated': None,
            'evaluation_count': 0,
            'evaluation_history': []
        }
    
    def _parse_neps_status(self, content: str) -> Dict:
        """Parse NePS status from text file."""
        status = {
            'started': None,
            'finished': None,
            'last_updated': None,
            'total_outer_folds': 0,
            'completed_outer_folds': 0,
            'outer_folds_progress': {}
        }
        
        # If this is a new file, don't parse anything yet
        if not content.strip():
            return status
        
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('==='):
                if 'NEPS TRAINING STATUS' in line:
                    current_section = 'neps'
                continue
            
            if current_section == 'neps':
                if line.startswith('Started:'):
                    value = line.split(':', 1)[1].strip()
                    if value != 'Not started' and value != 'None':
                        status['started'] = value
                elif line.startswith('Finished:'):
                    value = line.split(':', 1)[1].strip()
                    if value != 'Not finished' and value != 'None':
                        status['finished'] = value
                elif line.startswith('Last Updated:'):
                    value = line.split(':', 1)[1].strip()
                    if value != 'Never' and value != 'None':
                        status['last_updated'] = value
                elif line.startswith('Total Outer Folds:'):
                    try:
                        value = int(line.split(':', 1)[1].strip())
                        # Always set total_outer_folds, even if it's 0 (for debugging)
                        status['total_outer_folds'] = value
                    except ValueError:
                        pass
                elif line.startswith('Completed Outer Folds:'):
                    try:
                        value = int(line.split(':', 1)[1].strip())
                        status['completed_outer_folds'] = value
                    except ValueError:
                        pass
                elif line.startswith('Outer Fold'):
                    # Parse outer fold progress
                    try:
                        parts = line.split('|')
                        if len(parts) >= 2:
                            fold_info = parts[0].strip()
                            progress_info = parts[1].strip()
                            
                            fold_num = int(fold_info.split()[2])
                            
                            if 'COMPLETED' in progress_info:
                                status['outer_folds_progress'][fold_num] = {
                                    'status': 'completed',
                                    'inner_folds_completed': 0,
                                    'total_inner_folds': 0
                                }
                            else:
                                status['outer_folds_progress'][fold_num] = {
                                    'status': 'in_progress',
                                    'inner_folds_completed': 0,
                                    'total_inner_folds': 0,
                                    'inner_folds_status': {}
                                }
                    except (ValueError, IndexError):
                        pass
                elif line.startswith('  └─ Inner Fold'):
                    # Parse inner fold progress
                    try:
                        parts = line.split('|')
                        if len(parts) >= 2:
                            inner_fold_info = parts[0].strip()
                            status_info = parts[1].strip()
                            
                            # Extract inner fold number from "  └─ Inner Fold X"
                            inner_fold_num = int(inner_fold_info.split()[3])
                            
                            # Find the current outer fold (last one that was parsed)
                            current_outer_fold = max(status['outer_folds_progress'].keys()) if status['outer_folds_progress'] else None
                            
                            if current_outer_fold and 'inner_folds_status' not in status['outer_folds_progress'][current_outer_fold]:
                                status['outer_folds_progress'][current_outer_fold]['inner_folds_status'] = {}
                            
                            if current_outer_fold:
                                # Parse status and epoch info
                                if 'COMPLETED' in status_info:
                                    fold_status = 'completed'
                                    epoch = None
                                elif 'IN_PROGRESS' in status_info:
                                    fold_status = 'in_progress'
                                    # Extract epoch from "(Epoch X)"
                                    epoch_match = re.search(r'Epoch (\d+)', status_info)
                                    epoch = int(epoch_match.group(1)) if epoch_match else None
                                else:
                                    fold_status = 'unknown'
                                    epoch = None
                                
                                status['outer_folds_progress'][current_outer_fold]['inner_folds_status'][inner_fold_num] = {
                                    'status': fold_status,
                                    'last_updated': None,  # Will be set when loading
                                    'metrics': {},
                                    'current_epoch': epoch
                                }
                    except (ValueError, IndexError):
                        pass
        
        return status
    
    def _parse_quicktune_status(self, content: str) -> Dict:
        """Parse QuickTune status from text file."""
        status = {
            'started': None,
            'finished': None,
            'last_updated': None,
            'total_outer_folds': 0,
            'completed_outer_folds': 0,
            'outer_folds_progress': {}
        }
        
        # If this is a new file, don't parse anything yet
        if not content.strip():
            return status
        
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('==='):
                if 'QUICKTUNE STATUS' in line:
                    current_section = 'quicktune'
                continue
            
            if current_section == 'quicktune':
                if line.startswith('Started:'):
                    value = line.split(':', 1)[1].strip()
                    if value != 'Not started' and value != 'None':
                        status['started'] = value
                elif line.startswith('Finished:'):
                    value = line.split(':', 1)[1].strip()
                    if value != 'Not finished' and value != 'None':
                        status['finished'] = value
                elif line.startswith('Last Updated:'):
                    value = line.split(':', 1)[1].strip()
                    if value != 'Never' and value != 'None':
                        status['last_updated'] = value
                elif line.startswith('Total Outer Folds:'):
                    try:
                        value = int(line.split(':', 1)[1].strip())
                        status['total_outer_folds'] = value
                    except ValueError:
                        pass
                elif line.startswith('Completed Outer Folds:'):
                    try:
                        value = int(line.split(':', 1)[1].strip())
                        status['completed_outer_folds'] = value
                    except ValueError:
                        pass
                elif line.startswith('Outer Fold'):
                    # Parse outer fold progress
                    try:
                        parts = line.split('|')
                        if len(parts) >= 2:
                            fold_info = parts[0].strip()
                            progress_info = parts[1].strip()
                            
                            fold_num = int(fold_info.split()[2])
                            
                            if 'COMPLETED' in progress_info:
                                status['outer_folds_progress'][fold_num] = {
                                    'status': 'completed',
                                    'inner_folds_completed': 0,
                                    'total_inner_folds': 0
                                }
                            else:
                                status['outer_folds_progress'][fold_num] = {
                                    'status': 'in_progress',
                                    'inner_folds_completed': 0,
                                    'total_inner_folds': 0,
                                    'inner_folds_status': {}
                                }
                    except (ValueError, IndexError):
                        pass
                elif line.startswith('  └─ Inner Fold'):
                    # Parse inner fold progress
                    try:
                        parts = line.split('|')
                        if len(parts) >= 2:
                            inner_fold_info = parts[0].strip()
                            status_info = parts[1].strip()
                            
                            # Extract inner fold number from "  └─ Inner Fold X"
                            inner_fold_num = int(inner_fold_info.split()[3])
                            
                            # Find the current outer fold (last one that was parsed)
                            current_outer_fold = max(status['outer_folds_progress'].keys()) if status['outer_folds_progress'] else None
                            
                            if current_outer_fold and 'inner_folds_status' not in status['outer_folds_progress'][current_outer_fold]:
                                status['outer_folds_progress'][current_outer_fold]['inner_folds_status'] = {}
                            
                            if current_outer_fold:
                                # Parse status and epoch info
                                if 'COMPLETED' in status_info:
                                    fold_status = 'completed'
                                    epoch = None
                                elif 'IN_PROGRESS' in status_info:
                                    fold_status = 'in_progress'
                                    # Extract epoch from "(Epoch X)"
                                    epoch_match = re.search(r'Epoch (\d+)', status_info)
                                    epoch = int(epoch_match.group(1)) if epoch_match else None
                                else:
                                    fold_status = 'unknown'
                                    epoch = None
                                
                                status['outer_folds_progress'][current_outer_fold]['inner_folds_status'][inner_fold_num] = {
                                    'status': fold_status,
                                    'last_updated': None,  # Will be set when loading
                                    'metrics': {},
                                    'current_epoch': epoch
                                }
                    except (ValueError, IndexError):
                        pass
        
        return status
    
    def _parse_evaluation_status(self, content: str) -> Dict:
        """Parse evaluation status from text file."""
        status = {
            'started': None,
            'finished': None,
            'last_updated': None,
            'evaluation_count': 0,
            'evaluation_history': []
        }
        
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('==='):
                if 'EVALUATION STATUS' in line:
                    current_section = 'evaluation'
                continue
            
            if current_section == 'evaluation':
                # Parse new format
                if line.startswith('Started:'):
                    value = line.split(':', 1)[1].strip()
                    if value != 'Not started' and value != 'Never' and value != 'None':
                        status['started'] = value
                elif line.startswith('Last Updated:'):
                    value = line.split(':', 1)[1].strip()
                    if value != 'Never' and value != 'None':
                        status['last_updated'] = value
                elif line.startswith('Evaluation Count:'):
                    value = line.split(':', 1)[1].strip()
                    try:
                        status['evaluation_count'] = int(value)
                    except ValueError:
                        status['evaluation_count'] = 0
                elif line.startswith('Evaluation ') and ':' in line:
                    # Parse evaluation history entries
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        time_range = parts[1].strip()
                        if ' - ' in time_range:
                            started, finished = time_range.split(' - ', 1)
                            status['evaluation_history'].append({
                                'started': started.strip(),
                                'finished': finished.strip()
                            })
                # Parse old format for backward compatibility
                elif line.startswith('Evaluated:'):
                    value = line.split(':', 1)[1].strip()
                    if value != 'Never' and value != 'None':
                        status['last_updated'] = value
                        # Set started and finished to the same time for simple format
                        status['started'] = value
                        status['finished'] = value
                        status['evaluation_count'] = 1
        
        return status
    
    def update_main_progress(self, outer_fold: int, inner_folds_completed: int, total_inner_folds: int):
        """
        Update main training progress for a specific outer fold.
        
        Args:
            outer_fold (int): Outer fold number
            inner_folds_completed (int): Number of completed inner folds
            total_inner_folds (int): Total number of inner folds
        """
        if outer_fold not in self.main_status['outer_folds_progress']:
            self.main_status['outer_folds_progress'][outer_fold] = {
                'status': 'in_progress',
                'inner_folds_completed': 0,
                'total_inner_folds': total_inner_folds,
                'inner_folds_status': {}
            }
        
        self.main_status['outer_folds_progress'][outer_fold]['inner_folds_completed'] = inner_folds_completed
        self.main_status['outer_folds_progress'][outer_fold]['total_inner_folds'] = total_inner_folds
        
        # Check if this outer fold is completed
        if inner_folds_completed == total_inner_folds:
            self.main_status['outer_folds_progress'][outer_fold]['status'] = 'completed'
        
        # Update total completed outer folds
        completed_count = sum(1 for fold_info in self.main_status['outer_folds_progress'].values() 
                            if fold_info['status'] == 'completed')
        self.main_status['completed_outer_folds'] = completed_count
        
        self._save_main_status()
    
    def update_neps_progress(self, outer_fold: int, inner_folds_completed: int, total_inner_folds: int):
        """
        Update NePS training progress for a specific outer fold.
        (Backward compatibility method)
        """
        self.update_main_progress(outer_fold, inner_folds_completed, total_inner_folds)
    

    
    def set_total_outer_folds(self, total: int):
        """Set the total number of outer folds for the experiment."""
        if self.main_status['started'] is None:
            self.main_status['started'] = datetime.now().isoformat()
        
        self.main_status['total_outer_folds'] = total
        self._save_main_status()
    
    def mark_main_finished(self):
        """Mark the main experiment as finished."""
        self.main_status['finished'] = datetime.now().isoformat()
        self._save_main_status()
    
    def mark_neps_finished(self):
        """Mark the NePS experiment as finished. (Backward compatibility method)"""
        self.mark_main_finished()
    
    def log_evaluation(self):
        """
        Log that an evaluation was performed.
        """
        now = datetime.now().isoformat()
        
        # Read existing evaluation status to preserve history
        if self.evaluation_status_file.exists():
            try:
                with open(self.evaluation_status_file, 'r') as f:
                    content = f.read()
                    existing_status = self._parse_evaluation_status(content)
                    
                    if existing_status['evaluation_count'] > 0:
                        self.evaluation_status['evaluation_count'] = existing_status['evaluation_count']
                    if existing_status['evaluation_history']:
                        self.evaluation_status['evaluation_history'] = existing_status['evaluation_history']
            except Exception:
                pass
        
        # Set started time if this is the first evaluation
        if self.evaluation_status['started'] is None:
            self.evaluation_status['started'] = now
        
        # Update evaluation status
        self.evaluation_status['last_updated'] = now
        self.evaluation_status['evaluation_count'] += 1
        
        # Add evaluation with separate start and finish times
        self.evaluation_status['evaluation_history'].append({
            'started': now,
            'finished': now
        })
        
        # Keep only last 10 evaluations to avoid file bloat
        if len(self.evaluation_status['evaluation_history']) > 10:
            self.evaluation_status['evaluation_history'] = \
                self.evaluation_status['evaluation_history'][-10:]
        
        # Save the updated evaluation status to file
        self._save_evaluation_status()
    
    def _save_main_status(self):
        """Save main status to file."""
        self.main_status['last_updated'] = datetime.now().isoformat()
        
        # Format the status as a readable text file
        if self.experiment_type == "quicktune":
            status_text = self._format_quicktune_status_text()
        else:  # neps
            status_text = self._format_neps_status_text()
        
        with open(self.main_status_file, 'w') as f:
            f.write(status_text)
    
    def _save_neps_status(self):
        """Save NePS status to file. (Backward compatibility method)"""
        self._save_main_status()
    
    def _save_evaluation_status(self):
        """Save evaluation status to file."""
        self.evaluation_status['last_updated'] = datetime.now().isoformat()
        
        # Format the status as a readable text file
        status_text = self._format_evaluation_status_text()
        
        with open(self.evaluation_status_file, 'w') as f:
            f.write(status_text)
    
    def _format_evaluation_status_text(self) -> str:
        """Format the evaluation status data as a readable text file."""
        lines = []
        
        # Header
        lines.append("=" * 60)
        lines.append("EVALUATION STATUS")
        lines.append("=" * 60)
        
        # Simple format: just show when it was evaluated
        last_updated = self.evaluation_status.get('last_updated', 'Never')
        if last_updated != 'Never':
            lines.append(f"Evaluated: {last_updated}")
        else:
            lines.append("Evaluated: Never")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def _format_neps_status_text(self) -> str:
        """Format the NePS status data as a readable text file."""
        lines = []
        
        # Header
        lines.append("=" * 60)
        lines.append("NEPS TRAINING STATUS")
        lines.append("=" * 60)
        
        lines.append(f"Started: {self.main_status.get('started', 'Not started')}")
        lines.append(f"Finished: {self.main_status.get('finished', 'Not finished')}")
        lines.append(f"Last Updated: {self.main_status.get('last_updated', 'Never')}")
        lines.append("")
        
        total_outer = self.main_status['total_outer_folds']
        completed_outer = self.main_status['completed_outer_folds']
        
        lines.append(f"Total Outer Folds: {total_outer}")
        lines.append(f"Completed Outer Folds: {completed_outer}")
        if total_outer > 0:
            progress_percent = (completed_outer / total_outer) * 100
            lines.append(f"Overall Progress: {progress_percent:.1f}%")
        
        lines.append("")  
        
        # Individual outer fold progress
        total_outer = self.main_status['total_outer_folds']
        for fold_num in range(1, total_outer + 1):
            if fold_num in self.main_status['outer_folds_progress']:
                fold_info = self.main_status['outer_folds_progress'][fold_num]
                if fold_info['status'] == 'completed':
                    lines.append(f"Outer Fold {fold_num} | COMPLETED")
                elif fold_info['status'] == 'in_progress':
                    lines.append(f"Outer Fold {fold_num} | IN_PROGRESS")
                else:
                    lines.append(f"Outer Fold {fold_num} | {fold_info['status'].upper()}")
            else:
                # Outer fold hasn't started yet
                lines.append(f"Outer Fold {fold_num} | NOT STARTED")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def _format_quicktune_status_text(self) -> str:
        """Format the QuickTune status data as a readable text file."""
        lines = []
        
        # Header
        lines.append("=" * 60)
        lines.append("QUICKTUNE STATUS")
        lines.append("=" * 60)
        
        lines.append(f"Started: {self.main_status.get('started', 'Not started')}")
        lines.append(f"Finished: {self.main_status.get('finished', 'Not finished')}")
        lines.append(f"Last Updated: {self.main_status.get('last_updated', 'Never')}")
        lines.append("")
        
        total_outer = self.main_status['total_outer_folds']
        completed_outer = self.main_status['completed_outer_folds']
        
        lines.append(f"Total Outer Folds: {total_outer}")
        lines.append(f"Completed Outer Folds: {completed_outer}")
        if total_outer > 0:
            progress_percent = (completed_outer / total_outer) * 100
            lines.append(f"Overall Progress: {progress_percent:.1f}%")
        
        lines.append("")  
        
        # Individual outer fold progress
        total_outer = self.main_status['total_outer_folds']
        for fold_num in range(1, total_outer + 1):
            if fold_num in self.main_status['outer_folds_progress']:
                fold_info = self.main_status['outer_folds_progress'][fold_num]
                if fold_info['status'] == 'completed':
                    lines.append(f"Outer Fold {fold_num} | COMPLETED")
                elif fold_info['status'] == 'in_progress':
                    lines.append(f"Outer Fold {fold_num} | IN_PROGRESS")
                else:
                    lines.append(f"Outer Fold {fold_num} | {fold_info['status'].upper()}")
            else:
                # Outer fold hasn't started yet
                lines.append(f"Outer Fold {fold_num} | NOT STARTED")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    

    def get_evaluation_status(self) -> str:
        """Get the evaluation status."""
        if self.evaluation_status.get('last_updated'):
            return f"Evaluation done at: {self.evaluation_status['last_updated'][:19]}"
        else:
            return "Evaluation: Failed!"


class InnerFoldProgressLogger:
    """
    Logger for tracking progress of individual inner folds within each outer fold.
    Creates config-specific status files in config_X/outerfold_Y_status.txt format.
    """
    
    def __init__(self, pipeline_directory):
        """
        Initialize the inner fold progress logger.
        
        Args:
            pipeline_directory (str): Full pipeline directory path from NePS
        """
        from pathlib import Path
        
        # Extract outer fold number, config number, and base directory from pipeline path
        outer_fold, config_num, base_dir = self._extract_path_info(pipeline_directory)
        
        self.outer_fold = outer_fold
        self.config_num = config_num
        self.base_dir = Path(base_dir)
        self.status_dir = self.base_dir / "experiment_status" / f"config_{config_num}"
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.outer_fold_status = {}
    
    def _extract_path_info(self, pipeline_directory):
        """
        Extract outer fold number, config number, and base directory from pipeline directory.
        Supports both NePS and QuickTune directory structures.
        
        Args:
            pipeline_directory (str): Full pipeline directory path
            
        Returns:
            tuple: (outer_fold, config_num, base_dir) where outer_fold is 1-based and config_num is the config number
        """
        # Extract outer fold and config numbers from pipeline path
        pipeline_parts = str(pipeline_directory).split('/')
        outer_fold = None
        config_num = None
        
        for part in pipeline_parts:
            if part.startswith('cv_fold_'):
                outer_fold = int(part.split('_')[-1]) + 1
            elif part.startswith('config_'):
                config_num = int(part.split('_')[-1])
                break
            elif part.isdigit() and '/tuner/' in str(pipeline_directory):
                # QuickTune uses numeric config IDs in tuner directory
                config_num = int(part)
                break
        
        if outer_fold is None:
            outer_fold = 1
        
        if config_num is None:
            config_num = 1
        
        # Extract experiment base directory
        pipeline_str = str(pipeline_directory)
        if '/NePS_output/' in pipeline_str:
            base_dir = pipeline_str.split('/NePS_output/')[0]
        elif '/tuner/' in pipeline_str:
            # QuickTune structure: experiments/QuickTune/lipo/test_quicktune_1/seed_42/cv_fold_0/tuner/0/fold_0/...
            base_dir = pipeline_str.split('/cv_fold_')[0]
        else:
            base_dir = pipeline_str.split('/cv_fold_')[0]
        
        return outer_fold, config_num, base_dir
    
    def update_inner_fold_progress(self, inner_fold, status, epoch=None, total_inner_folds=None):
        """
        Update the progress of a specific inner fold within the current outer fold.
        
        Args:
            inner_fold (int): Which inner fold number (1-based)
            status (str): Current status ('in_progress', 'completed', etc.)
            epoch (int, optional): Current training epoch for progress tracking
            total_inner_folds (int, optional): Total number of inner folds for this outer fold
        """
        if self.outer_fold not in self.outer_fold_status:
            self.outer_fold_status[self.outer_fold] = {
                'inner_folds_status': {},
                'total_inner_folds': total_inner_folds or 0
            }
        
        self.outer_fold_status[self.outer_fold]['inner_folds_status'][inner_fold] = {
            'status': status,
            'last_updated': datetime.now().isoformat(),
            'current_epoch': epoch
        }
        
        if total_inner_folds is not None:
            self.outer_fold_status[self.outer_fold]['total_inner_folds'] = total_inner_folds
        
        self._save_outer_fold_status()
    
    def _save_outer_fold_status(self):
        """
        Save the current status of the outer fold to a human-readable text file.
        Creates files like: outerfold_1_status.txt, outerfold_2_status.txt, etc.
        """
        if self.outer_fold not in self.outer_fold_status:
            return  # Nothing to save if no status exists
        
        fold_info = self.outer_fold_status[self.outer_fold]
        outer_fold_file = self.status_dir / f"outerfold_{self.outer_fold}_status.txt"
        
        # Build the status file content
        lines = []
        lines.append("=" * 60)
        lines.append(f"NEPS CONFIG {self.config_num} - OUTER FOLD {self.outer_fold} STATUS")
        lines.append("=" * 60)
        lines.append(f"Config: {self.config_num}")
        lines.append(f"Outer Fold: {self.outer_fold}")
        lines.append(f"Started: {datetime.now().isoformat()}")
        lines.append(f"Last Updated: {datetime.now().isoformat()}")
        lines.append("")
        
        total_inner = fold_info['total_inner_folds']
        completed_inner = sum(1 for inner_info in fold_info['inner_folds_status'].values() 
                            if inner_info['status'] == 'completed')
        
        lines.append(f"Total Inner Folds: {total_inner}")
        lines.append(f"Completed Inner Folds: {completed_inner}")
        if total_inner > 0:
            progress_percent = (completed_inner / total_inner) * 100
            lines.append(f"Progress: {progress_percent:.1f}%")
        
        lines.append("")
        
        for inner_fold_num in sorted(fold_info['inner_folds_status'].keys()):
            inner_fold_info = fold_info['inner_folds_status'][inner_fold_num]
            status = inner_fold_info['status'].upper()
            epoch_info = f" (Epoch {inner_fold_info['current_epoch']})" if inner_fold_info.get('current_epoch') else ""
            
            lines.append(f"Inner Fold {inner_fold_num} | {status}{epoch_info}")
        
        lines.append("")
        lines.append("=" * 60)
        
        with open(outer_fold_file, 'w') as f:
            f.write("\n".join(lines)) 