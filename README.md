# Synth one-liner

![Run on Replicate](https://replicate.com/andreasjansson/synth-one-liner/badge)

In 2011 Viznut posted a blog post called ["Algorithmic symphonies from one line of code -- how and why?"](http://countercomplex.blogspot.com/2011/10/algorithmic-symphonies-from-one-line-of.html). He described a way to generate algorithmic 8-bit synth compositions in a single line of C. For example,

```c
main(t){for(t=0;;t++)putchar(t*(((t>>12)|(t>>8))&(63&(t>>4))));}
```

You'd compile that program and pipe it to your soundcard, and it would play an ever-evolving piece of noise music.

This is so fucking cool. That little equation `t*(((t>>12)|(t>>8))&(63&(t>>4)))` will tweak the bits of the value of the current time step as a kind of chaos process that actually sounds great!

I wanted to see if I could do that automatically.

## Automatic one-liners

Using [CodeLlama 7B](https://replicate.com/meta/codellama-7b) I make a prompt with [a bunch of one-liners from IRC](http://macumbista.net/wp-content/uploads/2016/11/music_formula_collection.txt).

I then use Llama.cpp's [grammar decoder](https://github.com/ggerganov/llama.cpp/issues/2364) to constrain the output to valid one-liners. Those one-liners are then evaluated and converted to audio.

It works surprisingly well, try it out here: https://replicate.com/andreasjansson/synth-one-liner
