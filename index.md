---
layout: page
title: tom's developer blog
tagline: a blog of how-tos covering linux, python, beaglebone and more
isHome: true
---
{% include JB/setup %}

## archive

<ul class="posts">
<table>
  {% for post in site.posts %}
    <tr>
    <td><span>{{ post.date | date_to_string }}</span> &raquo; <a href="{{ BASE_PATH }}{{ post.url }}">{{ post.title }}</a></td>
    </tr>
  {% endfor %}
</table>
</ul>

