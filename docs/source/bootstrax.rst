Bootstrax: XENONnT online processing manager
=============================================
The ``bootstrax`` script watches for new runs to appear from the DAQ, then starts a
strax process to process them.
``Restrax`` handles the output of bootstrax and stores it in a computing-friendly
manner (not too many, but few large files). The restrax documentation is on
If a run fails, it will retry it with
exponential backoff, each time waiting a little longer before retying.
After 10 failures, ``bootstrax`` stops trying to reprocess a run.
Additionally, every new time it is restarted it tries to process fewer plugins.
After a certain number of tries, it only reprocesses the raw-records.
Therefore a run that may fail at first may successfully be processed later. For example, if

You can run more than one ``bootstrax`` instance, but only one per machine.
If you start a second one on the same machine, it will try to kill the
first one.


Bootstrax philosophy
--------------------
Bootstrax has a crash-only / recovery first philosophy. Any error in
the core code causes a crash; there is no nice exit or mandatory
cleanup. Bootstrax focuses on recovery after restarts: before starting
work, we look for and fix any mess left by crashes.

This ensures that hangs and hard crashes do not require expert tinkering
to repair databases. Plus, you can just stop the program with ctrl-c
(or, in principle, pulling the machine's power plug) at any time.

Errors during run processing are assumed to be retry-able. We track the
number of failures per run to decide how long to wait until we retry;
only if a user marks a run as 'abandoned' (using an external system,
e.g. the website) do we stop retrying.


Mongo documents
----------------
Bootstrax records its status in a document in the '``bootstrax``' collection
in the runs db. These documents contain:

  - **host**: socket.getfqdn()
  - **time**: last time this ``bootstrax`` showed life signs
  - **state**: one of the following:
     - **busy**: doing something
     - **idle**: NOT doing something; available for processing new runs

Additionally, ``bootstrax`` tracks information with each run in the
'``bootstrax``' field of the run doc. We could also put this elsewhere, but
it seemed convenient. This field contains the following subfields:

  - **state**: one of the following:
        - **considering**: a ``bootstrax`` is deciding what to do with it
        - **busy**: a strax process is working on it
        - **failed**: something is wrong, but we will retry after some amount of time.
        - **abandoned**: ``bootstrax`` will ignore this run
  - **reason**: reason for last failure, if there ever was one (otherwise this field
    does not exists). Thus, it's quite possible for this field to exist (and
    show an exception) when the state is ``'done'``: that just means it failed
    at least once but succeeded later. Tracking failure history is primarily
    the DAQ log's responsibility; this message is only provided for convenience.
  - **n_failures**: number of failures on this run, if there ever was one
    (otherwise this field does not exist).
  - **next_retry**: time after which ``bootstrax`` might retry processing this run.
    Like 'reason', this will refer to the last failure.

Finally, ``bootstrax`` outputs the load on the eventbuilder machine(s)
whereon it is running to a collection in the DAQ database into the
capped collection 'eb_monitor'. This collection contains information on
what ``bootstrax`` is thinking of at the moment.

  - **disk_used**: used part of the disk whereto this ``bootstrax`` instance
    is writing to (in percent).

*Last updated 2023-02-14. Joran Angevaare*
