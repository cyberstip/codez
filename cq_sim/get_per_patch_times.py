import contextlib
import urllib


def Project(events):
  pass


def BotAttempt():
  pass


def ForEach(key_func, ):
  pass


def ExecCore(transformation):
  if isinstance(transformation, dict):
    return dict([(k, ExecCore(v)) for k,v in a.iteritems()])
  return transformation

project_timings = ForEach(Project,
    {
      'start_time':  PatchStart,
      'time_to_dispatch':  Diff(PatchStart, VerifierStart),
      'bot_trigger_map': BotTriggerMap,
      'bot_execution': ForEach(BotAttempt,
        {
          'execution_start': VerifierJobsBotStart,
          'execution_time': Diff(
            Max(Any(VerifierStart, VerifierRetry)), VerifierJobsBotFinish),
          'status_pickup_lag': Diff(
            VerifierJobsBotFinish, VerifierJobsOfBotFinish),
          'queue_and_schedulelag': Diff(
            Max(Any(VerifierStart, VerifierRetry, VerifierJobsBotParentEnd)),
            VerifierJobsBotStart),
          'execution_success': Exists(VerifierJobsUpdatePassed),
          'execution_was_retry': Exists(VerifierRetry),
        },
      ),
      'bot_finalization_lag': Diff(
        Max(VerifierJobsUpdate),
        Any(VerifierPass, VerifierFail)),
      'wait_on_tree': IfExists(PatchTreeClosed,
        Diff(PatchTreeClosed, PatchReadyForCommit)),
      'queue_for_commit': IfExists(PatchReadyForCommit,
        Diff(PatchReadyForCommit, PatchCommitting)),
      'commit_time': IfExists(PatchCommitting,
        Diff(PatchCommitting, PatchCommitted)),
    },
)

def get_stream():
  url = 'https://chromium-cq-status.appspot.com/query?count=10'


def extract_timings(timings=project_timings):
  stream = get_stream()

