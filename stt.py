# stt.py — works fully offline, no pythonnet, Py 3.12 compatible

import queue, threading, pythoncom, win32com.client

voice_in: queue.Queue[str] = queue.Queue(maxsize=1)

# Save queue at module level so handler can access
_queue_ref = voice_in

def _run():
    recog = win32com.client.Dispatch("SAPI.SpSharedRecognizer")
    ctx   = recog.CreateRecoContext()
    gram  = ctx.CreateGrammar()
    gram.DictationLoad()
    gram.DictationSetState(1)  # SPRS_ACTIVE

    class Handler:
        def OnRecognition(self, _sn, _sp, _rt, result):
            text = win32com.client.Dispatch(result).PhraseInfo.GetText()
            try:
                _queue_ref.put_nowait(text)
            except queue.Full:
                pass

    win32com.client.WithEvents(ctx, Handler)
    while True:
        pythoncom.PumpWaitingMessages()

threading.Thread(target=_run, daemon=True).start()
