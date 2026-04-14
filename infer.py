import dataclasses
import queue
import threading
import runpod
import ivrit
import logging

_SENTINEL = object()
MAX_BATCH_SIZE = 20

# Global variables to track the currently loaded model
current_model = None

def transcribe(job):
    engine = job['input'].get('engine', 'faster-whisper')
    model_name = job['input'].get('model', None)
    is_streaming = job['input'].get('streaming', False)

    if not engine in ['faster-whisper', 'stable-whisper']:
        yield { "error" : f"engine should be 'faster-whisper' or 'stable-whisper', but is {engine} instead." }

    if not model_name:
        yield { "error" : "Model not provided." }

    # Get the API key from the job input
    api_key = job['input'].get('api_key', None)

    # Extract transcribe_args from job input
    transcribe_args = job['input'].get('transcribe_args', None)

    # Validate that transcribe_args contains either blob or url
    if not transcribe_args:
        yield { "error" : "transcribe_args field not provided." }
    
    if not ('blob' in transcribe_args or 'url' in transcribe_args):
        yield { "error" : "transcribe_args must contain either 'blob' or 'url' field." }

    stream_gen = transcribe_core(engine, model_name, transcribe_args)

    if is_streaming:
        for entry in stream_gen:
            yield entry
    else:
        result = [entry for entry in stream_gen]
        yield { 'result' : result }

def transcribe_core(engine, model_name, transcribe_args):
    print('Transcribing...')

    global current_model

    different_model = (not current_model) or (current_model.engine != engine or current_model.model != model_name)

    if different_model:
        print(f'Loading new model: {engine} with {model_name}')
        current_model = ivrit.load_model(engine=engine, model=model_name, local_files_only=True)
    else:
        print(f'Reusing existing model: {engine} with {model_name}')

    q = queue.Queue()

    def on_progress(event):
        q.put({"type": "progress", "data": event})

    transcribe_args['on_progress'] = on_progress
    diarize = transcribe_args.get('diarize', False)

    def producer():
        try:
            if diarize:
                res = current_model.transcribe(**transcribe_args)
                segs = res['segments']
            else:
                transcribe_args['stream'] = True
                segs = current_model.transcribe(**transcribe_args)
            for s in segs:
                q.put({"type": "segments", "data": [dataclasses.asdict(s)]})
        except Exception as e:
            q.put(e)
        finally:
            q.put(_SENTINEL)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    try:
        while True:
            item = q.get()
            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                raise item

            batch = [item]
            while len(batch) < MAX_BATCH_SIZE and not q.empty():
                try:
                    more = q.get_nowait()
                    if more is _SENTINEL:
                        yield batch
                        return
                    if isinstance(more, Exception):
                        yield batch
                        raise more
                    batch.append(more)
                except queue.Empty:
                    break
            yield batch
    finally:
        thread.join()

runpod.serverless.start({"handler": transcribe, "return_aggregate_stream": True})

